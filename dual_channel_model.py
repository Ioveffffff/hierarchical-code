import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from utils.transformer import TransformerEncoderLayer


class EnhancedEntityEmbedding(nn.Module):
    """
    双通道实体表示模型

    通道1: Char CNN(完全可训练)
    通道2: BERT/RoBERTa(可微调，支持冻结前N层)

    架构流程:
    1. Char CNN + BERT → 双通道编码
    2. 局部CNN特征与文本token双向交叉注意力(可选)→ fused_emb (768维)
    3. instance_projector → z (128维) + normalize
    4. TransformerEncoder → h_hat (128维) + normalize
    5. cluster_projector → c (cluster_num)
    """

    def __init__(
            self,
            vocab,
            backbone_model_name,
            char_embed_dim=128,
            char_cnn_output_dim=512,
            backbone_dim=768,
            fusion_dim=768,
            instance_dim=128,
            cluster_num=None,
            use_sequence_features=False,
            use_bi_mcm_attn=False,
            fusion_method="gate",
            freeze_backbone_layers=2,
            max_seq_length=64,
            device="cuda"
    ):
        super(EnhancedEntityEmbedding, self).__init__()

        self.device = device
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.use_sequence_features = use_sequence_features
        self.use_bi_mcm_attn = use_bi_mcm_attn
        self.fusion_method = fusion_method
        self.fusion_dim = fusion_dim
        self.instance_dim = instance_dim
        self.max_seq_length = max_seq_length

        # 字符编码器
        self.char_embedding = nn.Embedding(self.vocab_size, char_embed_dim, padding_idx=0)

        if use_sequence_features:
            self.char_cnn = SequenceCharCNN(char_embed_dim, char_cnn_output_dim)
        else:
            self.char_cnn = SimpleCharCNN(char_embed_dim, char_cnn_output_dim)

        # 语义编码器
        print(f"加载可微调的backbone: {backbone_model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_model_name)
            self.backbone = AutoModel.from_pretrained(backbone_model_name)
        except Exception as e:
            print(f"加载失败: {e}")
            print(f"尝试使用本地路径或检查模型名称")
            raise

        self._freeze_backbone_layers(freeze_backbone_layers)

        self.backbone = self.backbone.to(device)

        # 双通道融合
        self.cross_attention = BidirectionalCrossAttention(
            char_cnn_dim=char_cnn_output_dim // 3,
            backbone_dim=backbone_dim,
            hidden_dim=256
        )
        if use_bi_mcm_attn and not use_sequence_features:
            raise ValueError("Local CNN cross-attention requires use_sequence_features=True")
        char_feature_dim = 256 if use_bi_mcm_attn else (char_cnn_output_dim // 3) * 3
        backbone_feature_dim = 256 if use_bi_mcm_attn else backbone_dim

        self.char_projection = nn.Sequential(
            nn.Linear(char_feature_dim, fusion_dim),
            nn.Dropout(0.1)
        )

        self.backbone_projection = nn.Sequential(
            nn.Linear(backbone_feature_dim, fusion_dim),
            nn.Dropout(0.1)
        )

        if fusion_method == "gate":
            self.fusion_gate = nn.Linear(fusion_dim * 2, fusion_dim)

        # 实例与簇投影
        self.instance_projector = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, instance_dim)
        )

        self.transformer_encoder = TransformerEncoderLayer(
            d_model=instance_dim,
            nhead=1,
            dim_feedforward=instance_dim * 2,
            dropout=0.1
        )

        if cluster_num is not None:
            self.cluster_projector = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(instance_dim, fusion_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(fusion_dim, fusion_dim),
                nn.ReLU(inplace=True),
                nn.Linear(fusion_dim, cluster_num)
            )
        else:
            self.cluster_projector = None

        self.to(device)

    def _freeze_backbone_layers(self, num_freeze_layers):
        """
        冻结backbone的前N层

        Args:
            num_freeze_layers: 0表示全部微调，12表示全部冻结(BERT-base有12层)
        """
        if num_freeze_layers == 0:
            print("   Backbone全部层可微调")
            for param in self.backbone.parameters():
                param.requires_grad = True
            return

        if num_freeze_layers >= 12:
            print("   Backbone全部层冻结")
            for param in self.backbone.parameters():
                param.requires_grad = False
            return

        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        for i in range(num_freeze_layers):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = False

        for i in range(num_freeze_layers, len(self.backbone.encoder.layer)):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True

        print(f"   冻结前{num_freeze_layers}层，微调后{12 - num_freeze_layers}层")

    def forward(self, entities, return_all=False):
        """
        Args:
            entities: list of strings
            return_all: 是否返回所有中间特征
        """
        batch_size = len(entities)

        char_indices = []
        for entity in entities:
            indices = [self.vocab.get(c, self.vocab.get('<UNK>', 1)) for c in entity]
            char_indices.append(indices)

        max_len = max(len(indices) for indices in char_indices)

        char_tensor = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)

        for i, indices in enumerate(char_indices):
            char_tensor[i, :len(indices)] = torch.tensor(indices, dtype=torch.long, device=self.device)

        char_emb = self.char_embedding(char_tensor)  # [B, L, D_char]
        if self.use_bi_mcm_attn:
            char_lengths = (char_tensor != 0).sum(dim=1)
            char_seq, char_mask = self.char_cnn.forward_sequence(char_emb, char_lengths)
        else:
            char_features = self.char_cnn(char_emb)  # [B, D_cnn]

        tokenized = self.tokenizer(
            entities,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )

        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)

        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        backbone_seq = backbone_outputs.last_hidden_state  # [B, N, D_backbone]

        if self.use_bi_mcm_attn:
            char_enhanced, backbone_enhanced = self.cross_attention(
                char_seq, backbone_seq, char_mask, attention_mask
            )
            char_proj = self.char_projection(char_enhanced)
            backbone_proj = self.backbone_projection(backbone_enhanced)
        else:
            backbone_emb = backbone_seq[:, 0, :]
            char_proj = self.char_projection(char_features)
            backbone_proj = self.backbone_projection(backbone_emb)

        char_proj = F.normalize(char_proj, p=2, dim=1)
        backbone_proj = F.normalize(backbone_proj, p=2, dim=1)

        if self.fusion_method == "gate":
            gate_input = torch.cat([char_proj, backbone_proj], dim=1)
            gate = self.fusion_gate(gate_input)
            gate = torch.sigmoid(gate)
            v = gate * char_proj + (1 - gate) * backbone_proj
        else:
            v = 0.3 * char_proj + 0.7 * backbone_proj

        v = F.normalize(v, p=2, dim=1)

        z = self.instance_projector(v)
        z_normalized = F.normalize(z, p=2, dim=1)

        h_hat, attn_weights = self.transformer_encoder(z_normalized)
        h_hat_normalized = F.normalize(h_hat, p=2, dim=1)

        if self.cluster_projector is not None:
            c = self.cluster_projector(h_hat_normalized)
        else:
            c = None

        if return_all:
            return {
                'v': v,
                'z': z,
                'z_normalized': z_normalized,
                'h_hat': h_hat,
                'h_hat_normalized': h_hat_normalized,
                'c': c,
                'attn_weights': attn_weights,
                'char_proj': char_proj,
                'backbone_proj': backbone_proj
            }
        else:
            return z_normalized


class SimpleCharCNN(nn.Module):
    def __init__(self, embed_dim=128, output_dim=512, kernel_sizes=(1, 2, 3)):
        super(SimpleCharCNN, self).__init__()
        print(f"Using SimpleCharCNN (kernels={kernel_sizes})")
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, output_dim // len(kernel_sizes), kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [B, L, D]
        x = x.transpose(1, 2)  # [B, D, L]
        conv_outputs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.adaptive_max_pool1d(out, 1).squeeze(2) for out in conv_outputs]
        out = torch.cat(pooled, dim=1)
        return self.dropout(out)


class SequenceCharCNN(nn.Module):
    def __init__(self, embed_dim=128, output_dim=512, kernel_sizes=(1, 2, 3)):
        super(SequenceCharCNN, self).__init__()
        print(f"Using SequenceCharCNN (dynamic kernels={kernel_sizes})")

        self.kernel_sizes = kernel_sizes
        self.out_per_kernel = output_dim // len(kernel_sizes)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=self.out_per_kernel,
                kernel_size=k
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        x: [B, L, D]
        return: [B, output_dim]
        """
        x = x.transpose(1, 2)  # [B, D, L]
        B, D, L = x.shape
        device = x.device

        outputs = []

        for conv, k in zip(self.convs, self.kernel_sizes):
            if L >= k:
                # 正常 n-gram 卷积
                y = conv(x)  # [B, C, L-k+1]
                y = F.relu(y)
                y = F.max_pool1d(y, y.size(2)).squeeze(2)  # [B, C]
            else:
                # 兜底：该 n-gram 对该样本不存在
                y = torch.zeros(
                    B,
                    self.out_per_kernel,
                    device=device
                )

            outputs.append(y)

        out = torch.cat(outputs, dim=1)  # [B, output_dim]
        return self.dropout(out)

    def forward_sequence(self, x, lengths):
        x = x.transpose(1, 2)
        batch_size, _, padded_length = x.shape
        features = []
        masks = []

        for conv, kernel_size in zip(self.convs, self.kernel_sizes):
            if padded_length >= kernel_size:
                feature = F.relu(conv(x)).transpose(1, 2)
            else:
                feature = x.new_zeros(batch_size, 1, self.out_per_kernel)

            valid_count = (lengths - kernel_size + 1).clamp(min=0)
            positions = torch.arange(feature.size(1), device=x.device)
            mask = positions.unsqueeze(0) < valid_count.unsqueeze(1)
            features.append(feature)
            masks.append(mask)

        return self.dropout(torch.cat(features, dim=1)), torch.cat(masks, dim=1)


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, char_cnn_dim=170, backbone_dim=768, hidden_dim=256, num_heads=4):
        super(BidirectionalCrossAttention, self).__init__()

        self.char_input = nn.Linear(char_cnn_dim, hidden_dim)
        self.backbone_input = nn.Linear(backbone_dim, hidden_dim)
        self.char_to_backbone = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.backbone_to_char = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm_c = nn.LayerNorm(hidden_dim)
        self.norm_b = nn.LayerNorm(hidden_dim)

    def forward(self, char_seq, backbone_seq, char_mask, backbone_mask):
        char_seq = self.char_input(char_seq)
        backbone_seq = self.backbone_input(backbone_seq)

        char_context, _ = self.char_to_backbone(
            char_seq, backbone_seq, backbone_seq,
            key_padding_mask=~backbone_mask.bool(), need_weights=False
        )
        backbone_context, _ = self.backbone_to_char(
            backbone_seq, char_seq, char_seq,
            key_padding_mask=~char_mask.bool(), need_weights=False
        )

        char_enhanced = self.norm_c(char_seq + char_context)
        backbone_enhanced = self.norm_b(backbone_seq + backbone_context)
        char_mask = char_mask.unsqueeze(-1)
        char_pooled = (char_enhanced * char_mask).sum(dim=1) / char_mask.sum(dim=1).clamp(min=1)
        return char_pooled, backbone_enhanced[:, 0, :]
