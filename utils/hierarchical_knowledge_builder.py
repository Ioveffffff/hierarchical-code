"""Build HCRL concept layers from CGCL clusters and ACRL representatives."""

import os
import json
import hashlib
import time
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm
import requests
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.metrics import silhouette_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.hcrl_leiden import leiden_candidates
from utils.hcrl_l1_catalog import catalog_candidates, parse_json_object
from utils.hcrl_name_embeddings import ConceptNameEncoder
from utils.hcrl_semantic_cache import SemanticReviewCache


# 发布或迁移到新数据集时，填写以下领域信息与示例。
# 这些占位符只影响 LLM 提示词和领域特定的宽泛名称过滤，不改变聚类算法。
# 在运行 HCRL 前，应结合数据集类别清单填写；不要让占位符直接发送给 LLM。
PROMPT_DOMAIN = "【填写数据集领域名称】"
PROMPT_TYPE_RULE = "【填写本领域必须区分的实体类型、属性类型和方法类型】"
PROMPT_FINE_EXAMPLE = "【填写细粒度术语及准确类名示例】"
PROMPT_MIDDLE_EXAMPLE = "【填写中层子概念及共同父概念示例】"
PROMPT_HIGH_EXAMPLE = "【填写高层子概念及共同父概念示例】"
PROMPT_PARENT_EXAMPLE = "【填写可合并的父子概念示例】"
PROMPT_WRONG_MERGE_EXAMPLE = "【填写表面相关但不应合并的示例】"
PROMPT_OVERBROAD_NAMES = {"【填写过宽类名示例一】", "【填写过宽类名示例二】"}


def _prompt_profile_values():
    return [PROMPT_DOMAIN, PROMPT_TYPE_RULE, PROMPT_FINE_EXAMPLE,
            PROMPT_MIDDLE_EXAMPLE, PROMPT_HIGH_EXAMPLE, PROMPT_PARENT_EXAMPLE,
            PROMPT_WRONG_MERGE_EXAMPLE, *sorted(PROMPT_OVERBROAD_NAMES)]


def _prompt_profile_signature():
    """Invalidate semantic decisions when dataset-specific prompt text changes."""
    payload = json.dumps(_prompt_profile_values(), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


# LLM 接口：生成上位概念
class OntologyLLM:
    def __init__(self, api_url=None, api_key=None, model_name="qwen-max", use_parallel=False, max_workers=3):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"} if api_key else None

        # 将本领域不可用的宽泛类名填入 PROMPT_OVERBROAD_NAMES。
        self.forbidden_generic_terms = {
            "相关技术", "其他", "杂项", "未分类", *PROMPT_OVERBROAD_NAMES
        }

        self.max_retry = 3  # 增加重试次数
        self.provisional_l1_names = {}  # LLM未给出具体名称时，记录待人工复核的暂定名

    def _get_abstract_level(self, layer: int) -> int:
        """动态计算抽象度"""
        return min(max(layer - 1, 0), 3)

    def _call_llm(self, prompt: str):
        """调用LLM API"""
        if not self.api_url or not self.api_key:
            return None
        if any("【填写" in value for value in _prompt_profile_values()):
            raise ValueError("运行 HCRL 前，请按具体数据集填写本文件开头的 PROMPT_* 占位符")
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 768,
            "temperature": 0.0
        }
        for attempt in range(3):
            try:
                r = requests.post(self.api_url, headers=self.headers,
                                  json=payload, timeout=90)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip()
                if r.status_code not in (429, 500, 502, 503, 504):
                    print(f"   LLM HTTP {r.status_code}", flush=True)
                    return None
                print(f"   LLM HTTP {r.status_code}，稍后重试", flush=True)
            except (requests.RequestException, KeyError, ValueError) as exc:
                print(f"   LLM调用失败: {type(exc).__name__}", flush=True)
            if attempt < 2:
                time.sleep(2 ** attempt)
        return None

    def choose_l1_category(self, cluster_id, terms, representatives, catalog):
        """Choose an exact catalog ID without seeing reference labels."""
        if not catalog:
            raise ValueError("L1 人工类别清单为空")
        options = [{"label": label, "name": name}
                   for label, name in sorted(catalog.items())]
        likely = catalog_candidates(terms, representatives, catalog)
        # L1 的可选类别由传入 catalog 决定，更新数据集时同步替换清单。
        prompt = (
            f"你是{PROMPT_DOMAIN}术语细分类审核员。只能从给出的类别清单中选一个编号，"
            f"不可新造名称。依据全部成员术语判断共同类别，并遵守类型规则：{PROMPT_TYPE_RULE}；"
            "代表术语仅帮助阅读，不能覆盖其他成员的冲突。"
            "即使簇内部混杂也选最能覆盖的现有类别，并标记 coherent=false，"
            "说明冲突术语；不要移动或删除术语。"
            "仅输出 JSON：label 为清单内整数，coherent 为布尔值，"
            "reason 为简短理由，conflicting_terms 为字符串数组。\n"
            f"簇编号：{cluster_id}\n"
            f"全部术语：{json.dumps(terms, ensure_ascii=False)}\n"
            f"代表术语：{json.dumps(representatives, ensure_ascii=False)}\n"
            "下面的相近名称只是字符检索提示，不是正确答案；须核对实体含义，"
            "也可选择完整清单中未列入提示的类别。\n"
            f"相近名称：{json.dumps(likely, ensure_ascii=False)}\n"
            f"可选类别：{json.dumps(options, ensure_ascii=False)}"
        )
        for attempt in range(3):
            raw = self._call_llm(prompt)
            if not raw:
                raise RuntimeError(f"L1-C{cluster_id} 受限命名服务无响应")
            answer = parse_json_object(raw)
            if (isinstance(answer, dict) and type(answer.get("label")) is int and
                    answer["label"] in catalog and
                    type(answer.get("coherent")) is bool and
                    isinstance(answer.get("conflicting_terms"), list)):
                return {"source_cluster_id": cluster_id,
                        "label": answer["label"],
                        "name": catalog[answer["label"]],
                        "coherent": answer["coherent"],
                        "reason": str(answer.get("reason", "")),
                        "conflicting_terms": [str(term) for term in answer["conflicting_terms"]]}
            prompt += "\n上次输出未提供清单内的整数编号和完整字段，请重新按 JSON 格式回答。"
        raise RuntimeError(f"L1-C{cluster_id} 未能返回有效的 {len(catalog)} 类编号及审查字段")

    def verify_l1_category(self, cluster_id, terms, representatives,
                           catalog, first_choice):
        """Independently compare a cached L1 choice with the full dataset catalog."""
        likely = catalog_candidates(terms, representatives, catalog)
        options = [{"label": label, "name": name}
                   for label, name in sorted(catalog.items())]
        prompt = (
            f"你是{PROMPT_DOMAIN}术语类别复核员。先前模型从 {len(catalog)} 类中选择了一个类别，"
            "但可能遗漏了清单中的更准确类别。独立检查全部术语与类别名称，"
            "尤其比较相近名称；若先前选择不符，改选更准确编号。"
            "相近名称只是检索提示，不限制最终答案；最终编号必须在完整清单中。"
            "coherent 判断簇内术语能否归入所选类别；"
            "category_fit 评分为 0=类别不符，1=仅相关，2=基本符合，3=准确对应。"
            "只输出一个 JSON 对象，字段为整数 label、布尔 coherent、"
            "整数 category_fit、字符串 reason、字符串数组 conflicting_terms；"
            "不要在 JSON 后输出解释。\n"
            f"簇编号：{cluster_id}\n"
            f"全部术语：{json.dumps(terms, ensure_ascii=False)}\n"
            f"代表术语：{json.dumps(representatives, ensure_ascii=False)}\n"
            f"先前选择：{json.dumps(first_choice, ensure_ascii=False)}\n"
            f"相近名称：{json.dumps(likely, ensure_ascii=False)}\n"
            f"完整清单：{json.dumps(options, ensure_ascii=False)}"
        )
        for _ in range(3):
            raw = self._call_llm(prompt)
            if not raw:
                raise RuntimeError(f"L1-C{cluster_id} 类别复核服务无响应")
            answer = parse_json_object(raw)
            if (type(answer.get("label")) is int and answer["label"] in catalog and
                    type(answer.get("coherent")) is bool and
                    type(answer.get("category_fit")) is int and
                    0 <= answer["category_fit"] <= 3 and
                    isinstance(answer.get("conflicting_terms"), list)):
                return {"source_cluster_id": cluster_id,
                        "label": answer["label"],
                        "name": catalog[answer["label"]],
                        "coherent": answer["coherent"],
                        "category_fit": answer["category_fit"],
                        "reason": str(answer.get("reason", "")),
                        "conflicting_terms": [str(term) for term in answer["conflicting_terms"]],
                        "initial_label": first_choice["label"],
                        "initial_name": first_choice["name"],
                        "name_changed": answer["label"] != first_choice["label"]}
            prompt += "\n上次未给出完整有效的单个 JSON 对象，请按字段要求重新回答。"
        raise RuntimeError(f"L1-C{cluster_id} 未能完成类别复核")

    def _build_prompt(self, entities: List[str], layer: int):
        """Legacy naming prompt; fill the domain placeholders above before use."""
        # 各层示例与抽象度应匹配当前数据集的概念体系。
        abstraction = self._get_abstract_level(layer)
        if layer == 1:
            return (
                f"你是{PROMPT_DOMAIN}术语分类专家。以下术语属于同一个细簇。"
                "依据实体类型或属性类型给出最具体、准确的类别名称。"
                f"本领域判别规则：{PROMPT_TYPE_RULE}。"
                f"细粒度命名示例：{PROMPT_FINE_EXAMPLE}。\n"
                f"代表术语：{json.dumps(entities, ensure_ascii=False)}\n"
                "只输出一个类别名称，不要解释。"
            )

        samples = "、".join(entities[:8])
        if len(entities) > 8:
            samples += f"等{len(entities)}个术语"
        level_rules = {
            0: ("具体类别名称", PROMPT_FINE_EXAMPLE),
            1: ("中等抽象度的共同上位名称", PROMPT_MIDDLE_EXAMPLE),
            2: ("较高抽象度但仍具体的方向名称", PROMPT_HIGH_EXAMPLE),
            3: ("高度抽象但保留领域特征的总体名称", PROMPT_HIGH_EXAMPLE),
        }
        target, example = level_rules[abstraction]
        return (
            f"你是{PROMPT_DOMAIN}领域的术语分类专家。"
            f"请为以下子概念生成{target}。\n"
            f"子概念列表：{samples}\n"
            f"本领域应区分的类型：{PROMPT_TYPE_RULE}。"
            f"对应层级的示例：{example}。"
            f"过宽名称示例：{sorted(PROMPT_OVERBROAD_NAMES)}。"
            "名称须覆盖全部子概念，不得用空泛词拼接不相容类别。"
            "只输出名称，不要解释。"
        )

    def _is_too_generic(self, concept: str, layer: int) -> bool:
        """
        判断概念名称是否过于泛化。
        """
        # 完全匹配黑名单
        if concept in self.forbidden_generic_terms:
            return True

        # L0-L1层：严格检查
        if layer <= 1:
            # 禁止以这些词结尾且前面没有具体修饰
            # 按领域调整过宽的名称后缀。
            bad_suffixes = ("工程", "工程学", "学", "概论", "基础")
            for suffix in bad_suffixes:
                if concept.endswith(suffix) and len(concept) <= len(suffix) + 2:
                    return True

        # L2及以上层：放宽但仍排除通用占位类名。
        else:
            # 只禁止最泛化的概念
            if concept in ["工程学", "技术学", "科学", "工程"]:
                return True

        return False

    def _postprocess(self, ans: str) -> str:
        """规范化LLM输出"""
        if not ans:
            return ""
        # 只读取第一条非空回答；不能把多行解释直接接到概念名后面。
        ans = next((line.strip() for line in str(ans).splitlines() if line.strip()), "")
        for separator in ("：", ":"):
            if separator in ans:
                prefix, value = ans.split(separator, 1)
                if prefix.strip() in {"名称", "分类名称", "概念名称", "技术方向", "技术领域"}:
                    ans = value.strip()
                    break
        # 清理常见的干扰字符
        ans = ans.replace(" ", "").strip("。；;，, ")
        ans = ans.replace("\"", "").replace("'", "").replace("、", "")

        # 清理可能的编号
        if ans and ans[0].isdigit():
            ans = ans[1:].lstrip(".")

        # 不截断概念名称；截断会把“技术方法”变成“技术方”。
        # 长度检查在 extract_concept 中完成，超长时要求 LLM 重新命名。
        return ans

    def extract_concept(self, entities: List[str], cid: int, layer: int) -> str:
        """
        生成并校验指定层级的概念名称。
        """
        if not entities:
            return f"概念_{layer}_{cid}"

        # 采样（最多8个代表性样本）
        sampled = entities[:min(8, len(entities))]

        # 构建初始prompt
        prompt = self._build_prompt(sampled, layer)
        saw_response = False

        # 多次重试
        for attempt in range(self.max_retry):
            raw = self._call_llm(prompt)
            saw_response = saw_response or bool(raw)
            concept = self._postprocess(raw)

            if not concept:
                continue

            if layer == 1 and ("待核查" in concept or "混合簇" in concept or
                               concept.startswith("细粒度簇_")):
                prompt += ("\n上次输出是占位名称。请根据代表术语给出最具体的暂定类别名称，"
                           "不要输出待核查、混合簇或编号；人工审核另行记录。")
                continue

            # 检查是否过于泛化
            if self._is_too_generic(concept, layer):
                print(f"   L{layer}-C{cid} 第{attempt+1}次生成过于泛化: {concept}")
                # 追加更严格的提示
                prompt += (f"\n\n【重要】上次输出\"{concept}\"过于泛化！"
                           f"请给出更具体的{PROMPT_DOMAIN}类别名称，并体现本领域的对象或属性。")
                continue

            # 检查长度
            if len(concept) < 3:
                print(f"   L{layer}-C{cid} 第{attempt+1}次生成过短: {concept}")
                prompt += f"\n\n【重要】上次输出\"{concept}\"过短！请给出完整的技术术语（4-10字）。"
                continue

            if len(concept) > 16:
                print(f"   L{layer}-C{cid} 第{attempt+1}次生成过长: {concept}")
                prompt += (f"\n\n【重要】上次输出\"{concept}\"超过16字。"
                           "请重新给出一个完整且简洁的技术概念名称，不要截断词语。")
                continue

            # 通过所有检查
            return concept

        # Fallback策略
        print(f"   L{layer}-C{cid} 经过{self.max_retry}次重试仍失败，使用Fallback")
        if layer == 1:
            if not saw_response:
                raise RuntimeError(f"L1-C{cid} 未收到 LLM 响应，不能凭空命名")
            # 已得到响应但名称仍不合格：使用明确的代表术语作暂定名，
            # 将这个节点单独送交事后人工审核，不在树中写占位类别。
            provisional = sampled[0]
            self.provisional_l1_names[cid] = {
                "name": provisional, "reason": "LLM未给出合格的具体类名"}
            return provisional

        # Fallback 1: 简化prompt
        simple_prompt = f"""
请用4-8个汉字总结这些{PROMPT_DOMAIN}术语的技术分类：{sampled[:3]}
要求：体现本领域的具体对象或属性，避免空泛名称。
只输出名称，不要解释。
"""
        raw = self._call_llm(simple_prompt)
        concept = self._postprocess(raw)
        if concept and 3 <= len(concept) <= 16 and not self._is_too_generic(concept, layer):
            print(f"   Fallback成功: {concept}")
            return concept

        # Fallback 2: 基于第一个样本的启发式命名
        if entities:
            head = entities[0]
            # 若需领域关键词兜底，请在此按数据集填写；
            # 默认仅使用术语前缀，不在公开代码中保留特定领域词表。
            return f"{head[:4]}类概念"

        # Fallback 3: 最终兜底
        return f"概念_{layer}_{cid}"

    def batch_extract_concepts(self, clusters_data: Dict[int, List[str]], layer: int) -> Dict[int, str]:
        """
        批量提取概念（支持并发）
        """
        if not self.use_parallel:
            # 串行模式
            concepts = {}
            for cid, samples in tqdm(clusters_data.items(), desc=f"L{layer} LLM概念生成", ncols=80):
                concepts[cid] = self.extract_concept(samples, cid, layer)
            return concepts

        # 并发模式
        concepts = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_cid = {
                executor.submit(self.extract_concept, samples, cid, layer): cid
                for cid, samples in clusters_data.items()
            }

            for future in tqdm(as_completed(future_to_cid), total=len(clusters_data),
                               desc=f"L{layer} LLM概念生成(并发)", ncols=80):
                cid = future_to_cid[future]
                try:
                    concepts[cid] = future.result()
                except Exception as e:
                    print(f"   Cluster {cid} 概念生成失败: {e}")
                    concepts[cid] = f"概念_{layer}_{cid}"

        return concepts

    def review_parent_merge(self, children: List[Dict], layer: int) -> Dict:
        """只核查算法候选父簇能否被具体命名；不调整子簇归属。"""
        # 上位类示例、不可合并反例及类型规则见文件开头的 PROMPT_*。
        if len(children) < 2:
            return {"accepted": False, "reason": "少于两个子簇", "parent_name": ""}

        if any("待核查" in child["name"] or child["name"].startswith("细粒度簇_")
               for child in children):
            return {"accepted": False, "reason": "存在尚未确认名称的子簇", "parent_name": ""}
        abstraction_rule = (
            "本层紧邻细簇，父类应是共同的实体类型或属性类型。"
            if layer <= 2 else
            f"父类可以是明确的上位概念或目录主题，例如{PROMPT_PARENT_EXAMPLE}；"
            "不同实体类型只有在共享具体对象或过程时才可同属一个主题，"
            f"不能仅因为都与{PROMPT_DOMAIN}有关就合并。"
        )
        prompt = (
            f"你是{PROMPT_DOMAIN}术语分类专家。聚类算法已经提出下面的候选父簇。"
            "你的任务仅是给父簇命名并核对名称是否覆盖每个子簇；"
            "不要修改成员、指定新的合并对象或为了凑层数创造概念。\n"
            f"层级：L{layer}；子簇：{json.dumps(children, ensure_ascii=False)}\n"
            "若子簇的 has_provisional_l1_descendant 为 true，其名称或下层名称未经确认；"
            "必须以全部子簇的代表术语核对共同上位概念，不能沿用可疑名称推断父类。"
            "无法核实包含关系时拒绝合并，且不要用宽泛名称掩盖不确定性。\n"
            f"层级要求：{abstraction_rule}\n"
            f"命名时遵守本领域的类型判别规则：{PROMPT_TYPE_RULE}。"
            "不能因修饰词相同就将不同实体或属性命名为同一具体概念。"
            "父名称须准确包含所有得 1–3 分的子簇，且比这些子簇逐步抽象；"
            "不相容的子簇给 0 分并退出该候选父簇。"
            "允许相关属性归入有明确对象限定的上位类，但不能伪称同一属性。"
            "若没有真实共同上位类，parent_name 留空，不要强行命名。"
            f"有共同上位类时给出具体且覆盖所有相容子簇的父名称，避免{sorted(PROMPT_OVERBROAD_NAMES)}等空泛名称；"
            "'与'、'及'只在构成准确的统一主题时使用，不能靠连接字拼接不相容类别。"
            f"不应合并的示例：{PROMPT_WRONG_MERGE_EXAMPLE}；可合并的示例：{PROMPT_PARENT_EXAMPLE}。"
            "若没有至少两个子簇有真实共同上位类则留空。\n"
            "逐个子簇给出 child_scores：0=父概念不包含，1=仅相关或勉强涵盖，"
            "2=父概念可以涵盖但较宽，3=明确准确地涵盖。"
            "有 0 分子簇时可为其余至少两个相容子簇给出具体父名称；"
            "0 分节点之后另找归属或单独传递。1 分子簇之后检查其他候选父概念。"
            "is_superordinate 只判断父名称是否为所有得 1–3 分子簇的共同上位类，"
            "不要求涵盖得 0 分的退出节点。"
            "仅输出 JSON 对象，键为 parent_name、child_scores（与子簇等长的整数数组）、"
            "is_superordinate（布尔值）和 reason。不要输出 accepted 或新的成员分配。"
        )
        last_reason = "名称无法覆盖全部子簇"
        any_response = False
        for attempt in range(2):
            raw = self._call_llm(prompt)
            any_response |= bool(raw)
            answer = parse_json_object(raw)
            if not answer:
                prompt += "\n上次响应不是有效 JSON，请按指定字段重新回答。"
                continue
            scores = answer.get("child_scores")
            if not (isinstance(scores, list) and len(scores) == len(children) and
                    all(type(value) is int and 0 <= value <= 3 for value in scores)):
                coverage = answer.get("child_coverage")
                scores = ([3 if value else 0 for value in coverage]
                          if not self.api_key and isinstance(coverage, list) and
                          len(coverage) == len(children) and
                          all(type(value) is bool for value in coverage) else [])
            name = self._postprocess(answer.get("parent_name", ""))
            last_reason = str(answer.get("reason") or last_reason)
            if answer.get("is_superordinate") is False or not name:
                if attempt == 0 and scores and sum(score >= 1 for score in scores) >= 2:
                    prompt += ("\n上次至少两个子簇得分不为0。请只针对这些相容子簇"
                               "重新判断并命名；0分子簇退出，不要求父名称涵盖它们。"
                               "若剩余子簇仍无共同上位概念或统一主题，再留空。")
                    continue
                return {"accepted": False, "reason": last_reason,
                        "parent_name": "", "child_scores": scores}
            if not (scores and sum(score >= 1 for score in scores) >= 2 and
                    answer.get("is_superordinate") is True and name):
                prompt += ("\n上次未给出能逐一覆盖子簇的共同上位名称。"
                           "若确实找不到，请将 parent_name 留空并说明原因。")
                continue
            invalid_connector = any(token in name for token in ("相关", "或", "/"))
            if (3 <= len(name) <= 16 and not invalid_connector and
                    not self._is_too_generic(name, layer)
                    and name not in PROMPT_OVERBROAD_NAMES
                    and "相关技术" not in name and name not in {child["name"] for child in children}):
                return {"accepted": True, "reason": last_reason,
                        "parent_name": name,
                        "child_scores": scores}
            prompt += ("\n上次父名称过宽、过短、与子名称相同，或不能表示统一主题。"
                       "请给出真正的共同上位类别；找不到则将 parent_name 留空。")
        if not any_response:
            raise RuntimeError("父簇命名服务无响应，不能当作语义拒绝")
        return {"accepted": False, "reason": last_reason, "parent_name": ""}

    def score_existing_parent(self, child: Dict, parent_name: str) -> int:
        """Assess one child against an already named candidate parent."""
        prompt = (
            f"判断已有父概念是否涵盖一个待归属的{PROMPT_DOMAIN}子概念。"
            "0=不包含，1=仅相关或勉强涵盖，2=可以涵盖但较宽，3=明确准确涵盖。"
            f"不得因为词面相近就忽略本领域类型规则：{PROMPT_TYPE_RULE}。"
            f"父概念：{parent_name}；子概念：{json.dumps(child, ensure_ascii=False)}。"
            "只输出 JSON：{\"score\":0到3的整数,\"reason\":\"简短理由\"}。"
        )
        for _ in range(2):
            raw = self._call_llm(prompt)
            if not raw:
                raise RuntimeError("候选父簇涵盖评分服务无响应")
            try:
                answer = json.loads(str(raw).strip().removeprefix("```json")
                                    .removesuffix("```").strip())
            except (TypeError, ValueError):
                answer = {}
            if isinstance(answer, dict) and type(answer.get("score")) is int and 0 <= answer["score"] <= 3:
                return answer["score"]
            prompt += "\n上次没有返回有效的 0–3 整数分数，请重新审核。"
        raise RuntimeError("候选父簇涵盖评分未返回有效 JSON")

    def score_parent_options(self, child: Dict, parent_names: List[str]) -> Dict[str, int]:
        """Review every supplied parent-child edge in batches of at most 12."""
        result = {}
        for offset in range(0, len(parent_names), 12):
            batch = parent_names[offset:offset + 12]
            prompt = (
                f"你是{PROMPT_DOMAIN}术语层次分类审核员。逐个判断下面每个已有父概念"
                "是否涵盖同一个待归属子概念。0=不包含，1=仅相关或勉强涵盖，"
                "2=可以涵盖但较宽，3=明确准确涵盖。"
                f"不能只因都与{PROMPT_DOMAIN}相关就给2或3；注意{PROMPT_TYPE_RULE}。"
                "必须逐项独立审核，返回与父概念列表等长、顺序相同的整数数组。"
                "只输出 JSON 对象：{\"scores\":[0,1,...]}。\n"
                f"子概念：{json.dumps(child, ensure_ascii=False)}\n"
                f"父概念列表：{json.dumps(batch, ensure_ascii=False)}"
            )
            for attempt in range(3):
                raw = self._call_llm(prompt)
                if not raw:
                    raise RuntimeError("批量父子涵盖评分服务无响应")
                scores = parse_json_object(raw).get("scores")
                if (isinstance(scores, list) and len(scores) == len(batch) and
                        all(type(score) is int and 0 <= score <= 3 for score in scores)):
                    result.update(zip(batch, scores))
                    break
                prompt += "\n上次分数数量或格式错误，请逐项审核后只返回指定 JSON。"
            else:
                raise RuntimeError(f"批量父子涵盖评分未返回 {len(batch)} 个有效分数")
        return result

    def resolve_upper_collision(self, name, first, second, occupied):
        prompt = (
            "两个上层候选父簇得到同一个名称。检查各自的直接子概念与代表术语。"
            "若它们确属同一上位概念且合并后名称仍涵盖全部子概念，same=true；"
            "否则 same=false，并为第二组给出准确、与已有名称不同的新名称。"
            "不得用编号或宽泛拼接词制造区别。只输出 JSON：same 布尔值、"
            "new_name 字符串、reason 字符串。\n"
            f"重名：{name}\n第一组：{json.dumps(first, ensure_ascii=False)}\n"
            f"第二组：{json.dumps(second, ensure_ascii=False)}\n"
            f"已有名称：{json.dumps(sorted(occupied), ensure_ascii=False)}"
        )
        for _ in range(3):
            raw = self._call_llm(prompt)
            if not raw:
                raise RuntimeError("上层同名核查服务无响应")
            try:
                answer = json.loads(str(raw).strip().removeprefix("```json")
                                    .removesuffix("```").strip())
            except (TypeError, ValueError):
                answer = {}
            if isinstance(answer, dict) and type(answer.get("same")) is bool:
                if answer["same"]:
                    return True, name, str(answer.get("reason", ""))
                new_name = self._postprocess(answer.get("new_name", ""))
                if (3 <= len(new_name) <= 16 and new_name not in occupied and
                        not self._is_too_generic(new_name, 2)):
                    return False, new_name, str(answer.get("reason", ""))
            prompt += "\n上次未给出有效的同类判断或可区分的新名称，请重新回答。"
        raise RuntimeError(f"上层同名 {name} 未能准确消解")


# 自适应簇代表学习模块（ACRL）
class AdaptiveClusterRepresentation:
    def __init__(self, alpha=0.6):
        self.alpha = alpha

    def compute_adaptive_center(self, cluster_embeddings: np.ndarray,
                                cluster_id: int, layer: int) -> tuple:
        """
        计算自适应簇中心。

        ACRL同时考虑空间紧致性和语义一致性：
        - 紧致性权重: exp(-d_i / sigma)
        - 一致性权重: exp(cos(v_i, mu)) / K
        - 自适应权重: alpha * w_compact + (1-alpha) * w_semantic
        """
        V = np.asarray(cluster_embeddings, dtype=np.float64)
        if len(V) == 1:
            return V[0], np.array([1.0])

        mu = np.mean(V, axis=0)
        distances = np.linalg.norm(V - mu, axis=1)
        sigma = float(np.std(distances))
        if sigma <= 1e-8:
            w_compact = np.full(len(V), 1.0 / len(V))
        else:
            logits = -distances / sigma
            w_compact = np.exp(logits - logits.max())
            w_compact = w_compact / w_compact.sum()

        mu_norm = mu / (np.linalg.norm(mu) + 1e-8)
        V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
        semantic_scores = np.dot(V_norm, mu_norm)
        w_semantic = np.exp(semantic_scores)
        w_semantic = w_semantic / w_semantic.sum()

        final_weights = self.alpha * w_compact + (1 - self.alpha) * w_semantic
        final_weights = final_weights / final_weights.sum()

        adaptive_center = np.sum(V * final_weights[:, None], axis=0)
        return adaptive_center, final_weights


# 层次一致性正则化停止准则（HCR-Stop）
class HierarchicalConsistencyStop:
    def __init__(
            self,
            use_hcr_stop=True,
            use_pcc=True,
            convergence_threshold=0.78,
            delta_j_threshold=0.05,
            pcc_low_threshold=0.30,
            pcc_high_threshold=0.98,
            min_layers=4,
            patience=1,
            delta_h_threshold=0.05,
            collapse_delta_h_threshold=0.015,
            no_abstraction_threshold=0.05,
            cumulative_compression_threshold=0.60,
            strong_abstraction_threshold=0.50,
            pcc_stable_threshold=0.80,
            pcc_valid_threshold=0.55,
            min_strong_abstraction_j=0.45
    ):
        """
        HCR-Stop for ontology tree construction.

        依据有效合并、层内结构质量和父子一致性停止。
        min_layers 与累计压缩阈值仅保留为兼容参数及诊断字段，
        不强制形成指定深度或压缩到指定簇数。
        """
        self.use_hcr_stop = use_hcr_stop
        self.use_pcc = use_pcc

        self.convergence_threshold = convergence_threshold
        self.delta_j_threshold = delta_j_threshold
        self.pcc_low_threshold = pcc_low_threshold
        self.pcc_high_threshold = pcc_high_threshold
        self.min_layers = min_layers
        self.patience = patience

        self.delta_h_threshold = delta_h_threshold
        self.collapse_delta_h_threshold = collapse_delta_h_threshold
        self.no_abstraction_threshold = no_abstraction_threshold

        # 保留累计压缩率供诊断，不作为扩层或停止的硬门槛。
        self.cumulative_compression_threshold = cumulative_compression_threshold

        # 历史配置字段，保留以兼容已有调用。
        self.strong_abstraction_threshold = strong_abstraction_threshold

        self.pcc_stable_threshold = pcc_stable_threshold
        self.pcc_valid_threshold = pcc_valid_threshold
        self.min_strong_abstraction_j = min_strong_abstraction_j

        self.history = {
            "entropy": [],
            "delta_h": [],
            "sc": [],
            "silhouette": [],
            "sc_sample_size": [],
            "singleton_fraction": [],
            "aggregation_coverage": [],
            "pcc": [],
            "pcc_floor": [],
            "pcc_floor_enforced": [],
            "J": [],
            "convergence_cutoff": [],
            "strong_abstraction_cutoff": [],
            "delta_J": [],
            "n_clusters": [],
            "input_count": [],
            "compression": [],
            "cumulative_compression": [],
            "stable_count": [],
            "no_abstraction_count": [],
            "low_delta_h_count": [],
            "stop_reason": [],
            "candidate_accepted": []
        }

        self.prev_entropy = None
        self.prev_pcc = None
        self.prev_J = None
        self.stable_count = 0
        self.no_abstraction_count = 0
        self.low_delta_h_count = 0

        # 第一次进入 HCR-Stop 时的 input_count，一般就是 L1 概念数。
        self.l1_concept_count = None

    def compute_parent_child_consistency(
            self,
            parent_centers: np.ndarray,
            child_centers: np.ndarray,
            parent_to_children: Dict[int, List[int]]
    ) -> float:
        """Diagnostic cosine similarity of each direct child to its parent."""
        parent_centers = np.asarray(parent_centers, dtype=np.float64)
        child_centers = np.asarray(child_centers, dtype=np.float64)

        scores = []
        for parent_id, child_ids in parent_to_children.items():
            parent_id = int(parent_id)
            child_ids = [int(x) for x in child_ids if 0 <= int(x) < len(child_centers)]

            if parent_id < 0 or parent_id >= len(parent_centers):
                raise ValueError(f"无效的父簇编号：{parent_id}")
            if len(child_ids) < 2:
                continue

            parent = parent_centers[parent_id]
            for child in child_centers[child_ids]:
                denominator = np.linalg.norm(child) * np.linalg.norm(parent)
                scores.append(float(np.dot(child, parent) / denominator)
                              if denominator > 1e-12 else 0.0)

        return float(np.clip(np.mean(scores), -1.0, 1.0)) if scores else 0.0

    def assess_candidate(self, X, labels, leaf_counts, pcc):
        """Score a complete candidate without changing HCR-Stop history."""
        labels = np.asarray(labels)
        leaf_counts = np.asarray(leaf_counts, dtype=float)
        if len(labels) != len(X) or len(leaf_counts) != len(X):
            raise ValueError("候选层的标签、术语计数与子节点数量不一致")
        count = len(np.unique(labels))
        totals = np.bincount(np.unique(labels, return_inverse=True)[1],
                             weights=leaf_counts)
        probabilities = totals / totals.sum()
        entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-10)))
        previous = self.prev_entropy
        delta_h = (max(0.0, previous - entropy) / max(abs(previous), 1e-10)
                   if previous is not None else 0.0)
        sc = (float(silhouette_score(X, labels, metric="cosine"))
              if 1 < count < len(labels) else 0.0)
        pcc = float(pcc)
        if not np.isfinite(pcc) or not 0 <= pcc <= 1:
            raise ValueError("PCC 必须来自有效的最终父子边分数")
        score = 0.35 * min(delta_h, 1.0) + 0.25 * ((sc + 1.0) / 2.0) + 0.40 * pcc
        return {"entropy": entropy, "delta_h": float(delta_h),
                "sc": sc, "pcc": pcc, "J": float(score),
                "n_clusters": count, "compression": 1 - count / len(labels)}

    def candidate_quality_declines(self, quality):
        """A smaller entropy gain alone is not cross-layer quality loss."""
        if self.prev_J is None or self.prev_pcc is None or not self.history["sc"]:
            return False
        tolerance = self.delta_j_threshold
        return (quality["J"] < self.prev_J - tolerance and
                quality["pcc"] < self.prev_pcc - tolerance and
                quality["sc"] < self.history["sc"][-1] - tolerance)

    def select_candidate(self, assessed):
        """Balance abstraction, structure, and coverage among comparable cuts."""
        best_j = max(row["quality"]["J"] for row in assessed)
        comparable = [row for row in assessed
                      if row["quality"]["J"] >= best_j - self.delta_j_threshold]
        if len(comparable) > 2:
            metrics = ("delta_h", "sc", "pcc")
            ranges = {key: (min(row["quality"][key] for row in comparable),
                            max(row["quality"][key] for row in comparable))
                      for key in metrics}

            def weakest_metric(row):
                normalized = []
                for key in metrics:
                    low, high = ranges[key]
                    normalized.append(
                        (row["quality"][key] - low) / (high - low)
                        if high - low > 1e-8 else 1.0)
                return min(normalized)

            return max(comparable, key=lambda row: (
                weakest_metric(row), row["quality"]["J"],
                row["quality"]["delta_h"], -row["resolution"]))
        return max(comparable, key=lambda row: (
            row["quality"]["delta_h"], row["quality"]["J"],
            row["quality"]["pcc"], -row["resolution"]))

    def abstraction_stalled(self, quality):
        return (quality["delta_h"] <= self.delta_h_threshold and
                self.low_delta_h_count + 1 >= self.patience)

    def adaptive_stopping_criterion(
            self,
            X: np.ndarray,
            labels: np.ndarray,
            layer: int,
            pcc: float = None,
            input_count: Optional[int] = None,
            leaf_counts: Optional[np.ndarray] = None,
            pcc_floor: Optional[float] = None,
            enforce_pcc_floor: bool = True
    ) -> tuple:
        """
        HCR-Stop 判定。

        Args:
            X: 当前被聚类的上一层中心向量。
            labels: 当前层对上一层中心的父节点分配标签。
            layer: 当前层编号，例如 L4 的 layer=4。
            pcc: 父簇内部子簇的留一向量一致性。
            input_count: 上一层概念数，默认 len(X)。用于计算局部压缩率。
            leaf_counts: 每个上一层概念覆盖的原始术语数；熵始终基于原始术语。

        Returns:
            should_stop, reason, weights, J
        """
        labels = np.asarray(labels)
        active_pcc_floor = (self.pcc_low_threshold if pcc_floor is None
                            else float(pcc_floor))
        unique_labels, inverse = np.unique(labels, return_inverse=True)
        n_clusters = len(unique_labels)
        input_count = int(input_count if input_count is not None else len(X))
        if len(labels) != len(X):
            raise ValueError("当前层标签数量与输入概念数不一致")
        if leaf_counts is None:
            leaf_counts = np.ones(len(labels), dtype=np.float64)
        leaf_counts = np.asarray(leaf_counts, dtype=np.float64)
        if (len(leaf_counts) != len(labels) or not np.isfinite(leaf_counts).all() or
                (leaf_counts <= 0).any()):
            raise ValueError("leaf_counts 必须为逐概念的正数原始术语计数")

        # 记录 L1 概念数量，仅用于累计压缩率的诊断。
        if self.l1_concept_count is None:
            self.l1_concept_count = max(input_count, 1)

        # 1. 熵变化 ΔH
        n_i = np.bincount(inverse, weights=leaf_counts)
        p_i = n_i / leaf_counts.sum()
        H = -np.sum(p_i * np.log(p_i + 1e-10))
        child_counts = np.bincount(inverse)
        singleton_fraction = float(np.count_nonzero(child_counts == 1) / len(labels))

        if self.prev_entropy is None:
            delta_h = 1.0
        else:
            delta_h = max(0.0, self.prev_entropy - H) / (abs(self.prev_entropy) + 1e-10)

        delta_h_norm = float(np.clip(delta_h, 0.0, 1.0))
        self.prev_entropy = H

        # 2. Silhouette
        sc_sample_size = min(len(labels), 2048)
        try:
            if n_clusters > 1 and n_clusters < len(labels):
                if len(labels) <= 2048:
                    SC = silhouette_score(X, labels, metric="cosine")
                else:
                    SC = silhouette_score(X, labels, metric="cosine",
                                          sample_size=sc_sample_size, random_state=0)
            else:
                SC = 0.0
        except Exception as exc:
            raise RuntimeError(f"L{layer} 的层内 Silhouette 无法计算") from exc

        # 3. PCC
        if self.use_pcc and pcc is not None:
            pcc_raw = float(np.clip(pcc, 0.0, 1.0))
            pcc_score = pcc_raw
        else:
            pcc_raw = 0.0
            pcc_score = 0.0

        if self.prev_pcc is None or pcc is None:
            delta_pcc = 0.0
        else:
            delta_pcc = pcc_raw - self.prev_pcc

        if pcc is not None:
            self.prev_pcc = pcc_raw

        # 4. 综合一致性分数 J
        if self.use_pcc:
            weights = {'ΔH': 0.35, 'SC': 0.25, 'PCC': 0.40}
        else:
            weights = {'ΔH': 0.55, 'SC': 0.45, 'PCC': 0.0}

        # 原先 (SC+1)/2 在原始 SC=0 时仍给予固定奖励。
        # 改用原始 SC，并将旧阈值平移 weights['SC']/2：SC=0 的判定边界
        # 不变，正/负 SC 则更直接地奖惩结构质量。
        convergence_cutoff = self.convergence_threshold - weights['SC'] / 2.0
        strong_abstraction_cutoff = self.min_strong_abstraction_j - weights['SC'] / 2.0
        J = (weights['ΔH'] * delta_h_norm +
             weights['SC'] * ((SC + 1.0) / 2.0) +
             weights['PCC'] * pcc_score)

        delta_j = 0.0 if self.prev_J is None else J - self.prev_J
        self.prev_J = J

        # 5. 局部压缩率
        # 当前层相对于上一层的压缩率。
        # 例如 110→108，compression=0.018，说明这一层几乎没抽象。
        compression = 1.0 - (float(n_clusters) / max(float(input_count), 1.0))
        compression = float(np.clip(compression, 0.0, 1.0))

        # 6. 累计压缩率（诊断，不设目标簇数）
        cumulative_compression = 1.0 - (
            float(n_clusters) / max(float(self.l1_concept_count), 1.0)
        )
        cumulative_compression = float(np.clip(cumulative_compression, 0.0, 1.0))

        print(
            f"   input_count={input_count}, n_clusters={n_clusters}, "
            f"compression={compression:.3f}, cumulative={cumulative_compression:.3f}"
        )
        print(f"   ΔH={delta_h_norm:.3f}, SC={SC:.3f}, PCC={pcc_raw:.3f}, "
              f"单成员比例={singleton_fraction:.1%}")
        print(
            f"   J={J:.3f} (ΔJ={delta_j:+.3f}), ΔPCC={delta_pcc:+.3f}, "
            f"stable={self.stable_count}/{self.patience}, "
            f"no_abs={self.no_abstraction_count}/{self.patience}"
        )

        # 7. 停止条件
        stop_conditions = {}

        if not self.use_hcr_stop:
            reason = "继续聚类（Fixed-Layers模式）"
            should_stop = False

        else:
            semantic_stable = (
                delta_h_norm <= self.delta_h_threshold and
                J >= convergence_cutoff and
                abs(delta_j) <= self.delta_j_threshold and
                (not self.use_pcc or pcc is None or pcc_raw >= self.pcc_stable_threshold)
            )
            self.stable_count = self.stable_count + 1 if semantic_stable else 0
            self.no_abstraction_count = (
                self.no_abstraction_count + 1
                if compression <= self.no_abstraction_threshold else 0
            )
            # 质量趋稳、单层压缩率低仅作诊断。候选已经全量数值筛选后，
            # 下一轮是否仍有可行合并才决定是否到顶；不能在本层提前停止。
            # Candidate comparisons and quality-decline stopping happen before
            # this accepted layer is committed; SC is not a semantic veto.
            should_stop = any(stop_conditions.values())
            reason = (f"满足停止条件: {', '.join(stop_conditions)}"
                      if should_stop else "继续聚类")

        # 8. 记录历史
        self.history["entropy"].append(float(H))
        self.history["delta_h"].append(float(delta_h_norm))
        self.history["sc"].append(float(SC))
        self.history["silhouette"].append(float(SC))
        self.history["sc_sample_size"].append(int(sc_sample_size))
        self.history["singleton_fraction"].append(singleton_fraction)
        self.history["aggregation_coverage"].append(1.0 - singleton_fraction)
        self.history["pcc"].append(float(pcc_raw))
        self.history["pcc_floor"].append(float(active_pcc_floor))
        self.history["pcc_floor_enforced"].append(bool(enforce_pcc_floor))
        self.history["J"].append(float(J))
        self.history["convergence_cutoff"].append(float(convergence_cutoff))
        self.history["strong_abstraction_cutoff"].append(float(strong_abstraction_cutoff))
        self.history["delta_J"].append(float(delta_j))
        self.history["n_clusters"].append(int(n_clusters))
        self.history["input_count"].append(int(input_count))
        self.history["compression"].append(float(compression))
        self.history["cumulative_compression"].append(float(cumulative_compression))
        self.history["stable_count"].append(int(self.stable_count))
        self.history["no_abstraction_count"].append(int(self.no_abstraction_count))
        self.low_delta_h_count = (self.low_delta_h_count + 1
                                  if delta_h_norm <= self.delta_h_threshold else 0)
        self.history["low_delta_h_count"].append(int(self.low_delta_h_count))
        self.history["stop_reason"].append(reason)
        self.history["candidate_accepted"].append(True)

        return should_stop, reason, weights, J

    def export_history(self, path: str):
        """导出 HCR-Stop 历史记录。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        print(f"指标历史已导出: {path}")

# 主类：层次化知识构建器（使用层次聚类）
class HierarchicalKnowledgeBuilder:
    """
    层次化簇代表学习模块（HCRL）：
    1. 语义聚类：ACRL 代表构图，Leiden 提出候选父簇
    2. 自适应簇中心计算：ACRL学习鲁棒簇代表
    3. 簇概念抽象生成：LLM生成由细到粗的概念标签
    4. 层次一致性判别：HCR-Stop基于ΔH、SC、PCC判断是否继续扩展
    """

    def __init__(self,
                 initial_embeddings: np.ndarray,
                 initial_entities: List[str],
                 initial_labels: np.ndarray,
                 output_dir: str = "./hierarchical_knowledge",
                 max_layers: int = 5,
                 llm_api_url: Optional[str] = None,
                 llm_api_key: Optional[str] = None,
                 llm_model_name: str = "qwen-max",
                 use_parallel_llm: bool = False,
                 acrl_alpha: float = 0.6,
                 use_hcr_stop: bool = True,
                 use_pcc: bool = True,
                 min_clusters_per_layer: int = 2,
                 min_aggregation_coverage: float = 0.0,
                 reference_labels: Optional[np.ndarray] = None,
                 hcr_min_layers: int = 4,
                 hcr_patience: int = 2,
                 leiden_neighbors: Optional[int] = None,
                 leiden_resolutions=None,
                 leiden_seeds=None,
                 l1_category_names: Optional[Dict[int, str]] = None,
                 name_embedding_model: str = "text-embedding-v4"):
        """
        Args:
            use_hcr_stop: 是否使用HCR-Stop动态停止（False=Fixed-Layers模式）
            use_pcc: 是否使用PCC指标（仅在use_hcr_stop=True时有效）
            min_clusters_per_layer: 每层自动切分后保留的最小簇数
            min_aggregation_coverage: 向量候选切分的最低合并覆盖率；默认 0，
                语义审核后允许无兄弟节点原样保留
            reference_labels: 计算DP时使用的人工参考类别；默认使用initial_labels
        """
        self.embeddings = np.asarray(initial_embeddings, dtype=np.float64)
        self.entities = initial_entities
        self.labels = np.asarray(initial_labels)
        self.reference_labels = np.asarray(reference_labels if reference_labels is not None else initial_labels)
        if (self.embeddings.ndim != 2 or len(self.embeddings) == 0 or
                len(self.embeddings) != len(self.entities) or
                len(self.labels) != len(self.entities) or
                len(self.reference_labels) != len(self.entities)):
            raise ValueError("嵌入、术语、初始伪标签和评价标签的样本数必须一致且非空")
        if (not np.isfinite(self.embeddings).all() or
                np.any(np.linalg.norm(self.embeddings, axis=1) <= 1e-12)):
            raise ValueError("余弦层次聚类要求每个术语向量有限且范数非零")
        self.output_dir = output_dir
        self.max_layers = max_layers
        self.min_clusters_per_layer = min_clusters_per_layer
        if not 0.0 <= min_aggregation_coverage <= 1.0:
            raise ValueError("min_aggregation_coverage 必须在 [0, 1] 内")
        self.min_aggregation_coverage = float(min_aggregation_coverage)
        # 按当前数据集诊断后传入 Leiden 参数，或用以下环境变量提供。
        # 公开代码不内置实验调出的具体取值。列表变量使用逗号分隔。
        if leiden_neighbors is None:
            leiden_neighbors = os.getenv("HCRL_LEIDEN_NEIGHBORS")
        if leiden_resolutions is None:
            leiden_resolutions = os.getenv("HCRL_LEIDEN_RESOLUTIONS")
        if leiden_seeds is None:
            leiden_seeds = os.getenv("HCRL_LEIDEN_SEEDS")
        if any(value is None for value in
               (leiden_neighbors, leiden_resolutions, leiden_seeds)):
            raise ValueError("请按数据集提供 Leiden 邻居数、分辨率候选和随机种子")
        if isinstance(leiden_resolutions, str):
            leiden_resolutions = leiden_resolutions.split(",")
        if isinstance(leiden_seeds, str):
            leiden_seeds = leiden_seeds.split(",")
        try:
            self.leiden_neighbors = int(leiden_neighbors)
            self.leiden_resolutions = tuple(float(value) for value in leiden_resolutions)
            self.leiden_seeds = tuple(int(value) for value in leiden_seeds)
        except (TypeError, ValueError) as exc:
            raise ValueError("Leiden 参数格式无效") from exc
        if (self.leiden_neighbors < 1 or not self.leiden_resolutions or
                not self.leiden_seeds or
                any(value <= 0 for value in self.leiden_resolutions)):
            raise ValueError("Leiden 邻居数和分辨率必须为正，候选列表不能为空")
        self.l1_category_names = dict(l1_category_names or {})
        # L1 类别数由当前数据集的清单决定，不在代码中写死。
        if (self.l1_category_names and
                (len(set(self.l1_category_names.values())) != len(self.l1_category_names) or
                 any(not name for name in self.l1_category_names.values()))):
            raise ValueError("L1 类别清单的名称须唯一且非空")

        self.llm = OntologyLLM(llm_api_url, llm_api_key, llm_model_name, use_parallel_llm)
        self.name_encoder = (ConceptNameEncoder(
            llm_api_url, llm_api_key, os.path.join(output_dir, "name_embedding_cache"),
            model=name_embedding_model) if llm_api_url and llm_api_key else None)
        self.acrl = AdaptiveClusterRepresentation(alpha=acrl_alpha)
        self.hcr_stop = HierarchicalConsistencyStop(
            use_hcr_stop=use_hcr_stop,
            use_pcc=use_pcc,
            min_layers=hcr_min_layers,
            patience=hcr_patience
        )

        self.prev_centers = None
        self.ontology = {}
        self.layer_linkages = {}
        self.leaf_clusters_by_layer = {}
        self.hierarchy_metrics = {}
        self.termination_reason = None
        self.quality_decline_count = 0
        self.leiden_reference_count = None

        os.makedirs(output_dir, exist_ok=True)

        print(f"层次化知识构建器初始化完成")
        print(f"   样本数: {len(self.entities)}")
        print(f"   初始向量维度: {self.embeddings.shape[1] if self.embeddings.ndim == 2 else 'unknown'}")
        print(f"   最大层数: {max_layers}")
        print(f"   聚类算法: ACRL 近邻图 + Leiden 多分辨率候选")
        print(f"   Leiden k={self.leiden_neighbors}, edge_mode=reciprocal_or_both, "
              f"weight=mean_percentile, resolutions={self.leiden_resolutions}, "
              f"seeds={self.leiden_seeds}")
        print(f"   最低聚合覆盖率: {self.min_aggregation_coverage:.0%}")
        print(f"   ACRL alpha: {acrl_alpha:.2f}")
        print(f"   HCR-Stop: {'启用' if use_hcr_stop else '禁用（Fixed-Layers模式）'}")
        print(f"   PCC指标: {'启用' if use_pcc else '禁用'}")
        print("   各层使用同一 Leiden 候选策略；分辨率随节点数缩放")


    @staticmethod
    def _leiden_supports_reassignment(child_id, target_children, candidate, partitions):
        """Require a proposed community to connect the child to the target parent."""
        target_children = set(target_children)
        for alternative in partitions:
            if alternative["resolution"] < candidate["resolution"]:
                continue
            for group in alternative["groups"]:
                if child_id in group and target_children.intersection(group):
                    return True
        return False

    def _plan_leiden_groups(self, X, names, examples, flags, layer):
        """Broaden the resolution search only when no semantic merge survives."""
        reference = self.leiden_reference_count or len(X)
        resolution_scale = len(X) / reference
        resolutions = tuple(gamma * resolution_scale
                            for gamma in self.leiden_resolutions)
        attempts = []
        all_decisions = []
        print(f"   L{layer} Leiden 分辨率按节点数缩放："
              f"{len(X)}/{reference}，本层 {resolutions}", flush=True)
        while True:
            groups, decisions, info = self._plan_leiden_groups_once(
                X, names, examples, flags, layer, resolutions)
            all_decisions.extend(decisions)
            attempts.append({"resolutions": list(resolutions),
                             "candidate_count": info["candidate_count"],
                             "reviewed_group_count": info["llm_review_count"],
                             "valid_merge_count": info.get("accepted_merge_count", 0)})
            info["resolution_search"] = attempts
            info["resolution_scale"] = resolution_scale
            info["resolution_reference_count"] = reference
            if (groups or info["min_community_count"] <= info["component_count"]
                    or min(resolutions) == 0):
                return groups, all_decisions, info
            if min(resolutions) <= 1e-6:
                resolutions = (0.0,)
            else:
                resolutions = tuple(gamma / 2 for gamma in resolutions)
            print(f"   L{layer} 当前分辨率未形成有效父簇，继续检查 "
                  f"{resolutions}", flush=True)

    def _plan_leiden_groups_once(self, X, names, examples, flags, layer,
                                 resolutions):
        """Review each complete Leiden cut before HCR-Stop chooses a layer."""
        if self.name_encoder is None:
            raise RuntimeError("双向量 Leiden 需要名称嵌入 API")
        print(f"   L{layer} 编码 {len(names)} 个概念名称并生成 Leiden 候选层...",
              flush=True)
        name_vectors = self.name_encoder.encode(names)
        partitions, diagnostics = leiden_candidates(
            X, name_vectors, neighbors=self.leiden_neighbors,
            resolutions=resolutions, seeds=self.leiden_seeds)
        candidate_edges = diagnostics.pop("candidate_edges", [])
        diagnostics["min_community_count"] = min(
            (len(candidate["groups"]) for candidate in partitions),
            default=len(X))
        cache = SemanticReviewCache(
            os.path.join(self.output_dir, f"l{layer}_semantic_cache_v3.jsonl"),
            X, names, examples, flags, layer,
            f"{self.llm.model_name}:{_prompt_profile_signature()}")
        review_cache = cache.groups
        edge_cache = cache.edges
        decisions = []
        logged_groups = set()
        assessed = []
        leaf_counts = np.array([len(self._current_leaf_indices[i])
                                for i in range(len(X))], dtype=float)
        print(f"   Leiden 完整候选层={len(partitions)}；已缓存组审核="
              f"{len(review_cache)}、父子边评分={len(edge_cache)}", flush=True)
        pending_groups = {}
        for candidate in partitions:
            for indices in candidate["groups"]:
                key = tuple(indices)
                if len(key) >= 2 and key not in review_cache and key not in pending_groups:
                    pending_groups[key] = [
                        {"name": names[i],
                         "representative_terms": examples[i],
                         "has_provisional_l1_descendant": bool(flags[i])}
                        for i in key]
        print(f"   L{layer} 待审核的不重复候选组={len(pending_groups)}",
              flush=True)
        if pending_groups:
            failures = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(self.llm.review_parent_merge, children, layer): key
                           for key, children in pending_groups.items()}
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"L{layer} 候选父簇语义审核", ncols=80):
                    key = futures[future]
                    try:
                        decision = future.result()
                    except Exception as exc:
                        failures.append((key, exc))
                    else:
                        cache.record_group(key, decision)
            if failures:
                key, exc = failures[0]
                raise RuntimeError(
                    f"L{layer} 有 {len(failures)} 组语义审核失败；已完成的组保存在断点中，"
                    f"首个失败组={key}") from exc
        for candidate_index, candidate in enumerate(partitions, 1):
            accepted = []
            free = set(range(len(X)))
            scores = {}
            rejected_from = {}
            for indices in candidate["groups"]:
                if len(indices) < 2:
                    continue
                key = tuple(indices)
                if key not in logged_groups:
                    decisions.append({"candidate_source": "leiden_community",
                                      "child_ids": indices, **review_cache[key]})
                    logged_groups.add(key)
                decision = review_cache[key]
                if not decision.get("accepted"):
                    continue
                included = [i for i, score in zip(indices, decision["child_scores"])
                            if score >= 1]
                if len(included) < 2:
                    continue
                parent = {"children": included, "name": decision["parent_name"]}
                accepted.append(parent)
                for child_id in indices:
                    if child_id not in included:
                        rejected_from[child_id] = parent
                scores.update((i, score) for i, score in
                              zip(indices, decision["child_scores"]) if i in included)
                free.difference_update(included)
            source_by_child = {child_id: source for source in accepted
                               for child_id in source["children"]}
            original_children = {id(target): tuple(target["children"])
                                 for target in accepted}
            to_score = sorted(set(free) | {child_id for child_id, score in scores.items()
                                           if score == 1})
            requests_to_make = {}
            for child_id in to_score:
                parent_names = list(dict.fromkeys(
                    target["name"] for target in accepted
                    if (target is not source_by_child.get(child_id) and
                        target is not rejected_from.get(child_id) and
                        self._leiden_supports_reassignment(
                            child_id, original_children[id(target)],
                            candidate, partitions))))
                missing = [parent for parent in parent_names
                           if (child_id, parent) not in edge_cache]
                if missing:
                    requests_to_make[child_id] = missing
            if requests_to_make:
                failures = []
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {}
                    for child_id, parent_names in requests_to_make.items():
                        child = {"name": names[child_id],
                                 "representative_terms": examples[child_id],
                                 "has_provisional_l1_descendant": bool(flags[child_id])}
                        for offset in range(0, len(parent_names), 12):
                            batch = parent_names[offset:offset + 12]
                            future = executor.submit(
                                self.llm.score_parent_options, child, batch)
                            futures[future] = (child_id, batch)
                    for future in tqdm(as_completed(futures), total=len(futures),
                                       desc=f"L{layer} 候选 {candidate_index} 父子边审核",
                                       ncols=80):
                        child_id, batch = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            failures.append((child_id, batch, exc))
                        else:
                            for parent_name, score in result.items():
                                cache.record_edge(child_id, parent_name, score)
                if failures:
                    child_id, batch, exc = failures[0]
                    raise RuntimeError(
                        f"L{layer} 有 {len(failures)} 批父子边审核失败；已完成评分保存在断点中，"
                        f"首个失败子节点={child_id}、候选父类={batch}") from exc
                print(f"   L{layer} 候选 {candidate_index}/{len(partitions)}："
                      f"批量审核 {len(requests_to_make)} 个子节点，"
                      f"累计父子边评分 {len(edge_cache)}", flush=True)
            for source in list(accepted):
                for child_id in list(source["children"]):
                    if scores[child_id] != 1:
                        continue
                    child = {"name": names[child_id],
                             "representative_terms": examples[child_id],
                             "has_provisional_l1_descendant": bool(flags[child_id])}
                    for target in accepted:
                        if target is source:
                            continue
                        if not self._leiden_supports_reassignment(
                                child_id, original_children[id(target)],
                                candidate, partitions):
                            continue
                        edge_key = (child_id, target["name"])
                        if edge_key not in edge_cache:
                            raise RuntimeError(f"缺失已计划的父子边评分：{edge_key}")
                        if edge_cache[edge_key] >= 2 and edge_cache[edge_key] > scores[child_id]:
                            source["children"].remove(child_id)
                            target["children"].append(child_id)
                            scores[child_id] = edge_cache[edge_key]
                            decisions.append({"candidate_source": "alternative_parent",
                                              "child_ids": [child_id],
                                              "parent_name": target["name"],
                                              "child_scores": [scores[child_id]]})
                            break
            # Rejected children can still join another named candidate parent.
            for child_id in sorted(list(free)):
                child = {"name": names[child_id],
                         "representative_terms": examples[child_id],
                         "has_provisional_l1_descendant": bool(flags[child_id])}
                best_score, best_target = 0, None
                for target in accepted:
                    if target is rejected_from.get(child_id):
                        continue
                    if not self._leiden_supports_reassignment(
                            child_id, original_children[id(target)],
                            candidate, partitions):
                        continue
                    edge_key = (child_id, target["name"])
                    if edge_key not in edge_cache:
                        raise RuntimeError(f"缺失已计划的父子边评分：{edge_key}")
                    if edge_cache[edge_key] > best_score:
                        best_score, best_target = edge_cache[edge_key], target
                if best_target is not None and best_score >= 2:
                    best_target["children"].append(child_id)
                    scores[child_id] = best_score
                    free.remove(child_id)
                    decisions.append({"candidate_source": "rejected_child_alternative",
                                      "child_ids": [child_id],
                                      "parent_name": best_target["name"],
                                      "child_scores": [best_score]})
            groups = []
            for group in accepted:
                if len(group["children"]) >= 2:
                    groups.append((sorted(group["children"]), group["name"], False))
                else:
                    free.update(group["children"])
                    for child_id in group["children"]:
                        scores.pop(child_id, None)
            groups.extend(([i], names[i], True) for i in sorted(free))
            groups.sort(key=lambda row: min(row[0]))
            groups = self._resolve_upper_names(groups, names, examples, scores,
                                               decisions, cache)
            if sorted(i for group, _, _ in groups for i in group) != list(range(len(X))):
                raise ValueError("Leiden 候选层未唯一覆盖全部子节点")
            if not any(len(group) > 1 for group, _, _ in groups):
                continue
            labels = np.empty(len(X), dtype=int)
            for parent_id, (children, _, _) in enumerate(groups):
                labels[children] = parent_id
            pcc = self._semantic_pcc(groups, scores)
            quality = self.hcr_stop.assess_candidate(X, labels, leaf_counts, pcc)
            assessed.append({"groups": groups, "scores": scores,
                             "resolution": candidate["resolution"],
                             "seed": candidate["seed"], "quality": quality})
            print(f"   L{layer} 候选 {candidate_index}/{len(partitions)} "
                  f"γ={candidate['resolution']:g}：有效父簇="
                  f"{sum(len(group) > 1 for group, _, _ in groups)}，"
                  f"PCC={pcc:.3f}，J={quality['J']:.3f}", flush=True)
        self._refine_sparse_layer(
            X, names, examples, flags, layer, leaf_counts, partitions,
            candidate_edges, review_cache, cache, assessed, decisions)
        if not assessed:
            diagnostics.update(llm_review_count=len(review_cache),
                               assessed_candidates=[])
            return [], decisions, diagnostics
        best = self.hcr_stop.select_candidate(assessed)
        diagnostics.update(
            llm_review_count=len(review_cache),
            assessed_candidates=[{"resolution": row["resolution"],
                                  "seed": row["seed"],
                                  "source": row.get("source", "leiden_cut"),
                                  **row["quality"]}
                                 for row in assessed],
            selected_resolution=best["resolution"],
            selected_seed=best["seed"],
            selected_source=best.get("source", "leiden_cut"),
            selected_quality=best["quality"],
            accepted_merge_count=sum(len(group) > 1 for group, _, _ in best["groups"]),
            pass_through_count=sum(len(group) == 1 for group, _, _ in best["groups"]),
            semantic_scores_by_child=best["scores"])
        return best["groups"], decisions, diagnostics

    def _refine_sparse_layer(self, X, names, examples, flags, layer, leaf_counts,
                             partitions, candidate_edges, review_cache, cache,
                             assessed, decisions):
        """Review graph-supported subgroups when full cuts yield little abstraction."""
        if assessed:
            best = self.hcr_stop.select_candidate(assessed)
            accepted = sum(len(children) > 1 for children, _, _ in best["groups"])
            if (best["quality"]["delta_h"] > self.hcr_stop.delta_h_threshold
                    and accepted > 2):
                return
        rejected = [set(group) for candidate in partitions
                    for group in candidate["groups"] if len(group) > 1 and
                    not review_cache[tuple(group)].get("accepted")]
        pairs = sorted({tuple(sorted((left, right)))
                        for left, right in candidate_edges
                        if any(left in group and right in group for group in rejected)})
        pending = [pair for pair in pairs if pair not in review_cache]
        if pending:
            failures = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                for pair in pending:
                    children = [{"name": names[i],
                                 "representative_terms": examples[i],
                                 "has_provisional_l1_descendant": bool(flags[i])}
                                for i in pair]
                    futures[executor.submit(
                        self.llm.review_parent_merge, children, layer)] = pair
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"L{layer} 图边子组补查", ncols=80):
                    pair = futures[future]
                    try:
                        cache.record_group(pair, future.result())
                    except Exception as exc:
                        failures.append((pair, exc))
            if failures:
                pair, exc = failures[0]
                raise RuntimeError(f"L{layer} 图边子组补查失败：{pair}") from exc
        options = {}
        for row in assessed:
            for children, name, _ in row["groups"]:
                if len(children) < 2:
                    continue
                child_ids = tuple(children)
                values = tuple(row["scores"][child] for child in child_ids)
                if (child_ids not in options or
                        np.mean(values) > np.mean(options[child_ids]["scores"])):
                    options[child_ids] = {"name": name, "scores": values}
        for pair in pairs:
            decision = review_cache[pair]
            if not decision.get("accepted"):
                continue
            values = decision.get("child_scores", [])
            if len(values) != 2 or min(values) < 1:
                continue
            if (pair not in options or
                    np.mean(values) > np.mean(options[pair]["scores"])):
                options[pair] = {"name": decision["parent_name"],
                                 "scores": tuple(values)}
            decisions.append({"candidate_source": "leiden_graph_edge",
                              "child_ids": list(pair), **decision})
        if not options:
            return
        choices = list(options.items())
        membership = np.zeros((len(X), len(choices)))
        utility = np.empty(len(choices))
        total = leaf_counts.sum()
        for index, (children, option) in enumerate(choices):
            membership[list(children), index] = 1
            probabilities = leaf_counts[list(children)] / total
            mass = probabilities.sum()
            entropy_gain = (mass * np.log(mass) -
                            np.sum(probabilities * np.log(probabilities)))
            utility[index] = entropy_gain * np.mean(option["scores"]) / 3
        solution = milp(
            c=-utility, integrality=np.ones(len(choices)),
            bounds=Bounds(np.zeros(len(choices)), np.ones(len(choices))),
            constraints=LinearConstraint(membership, -np.inf, 1))
        if not solution.success:
            raise RuntimeError(f"L{layer} 不重叠候选组合求解失败：{solution.message}")
        selected = [choices[index] for index, value in enumerate(solution.x)
                    if value > 0.5]
        if not selected:
            return
        used = set()
        scores = {}
        groups = []
        for children, option in selected:
            used.update(children)
            scores.update(zip(children, option["scores"]))
            groups.append((list(children), option["name"], False))
        groups.extend(([index], names[index], True)
                      for index in range(len(X)) if index not in used)
        groups.sort(key=lambda row: min(row[0]))
        groups = self._resolve_upper_names(groups, names, examples, scores,
                                           decisions, cache)
        if sorted(index for children, _, _ in groups for index in children) != list(range(len(X))):
            raise ValueError("补查候选层没有唯一覆盖全部子节点")
        labels = np.empty(len(X), dtype=int)
        for parent_id, (children, _, _) in enumerate(groups):
            labels[children] = parent_id
        quality = self.hcr_stop.assess_candidate(
            X, labels, leaf_counts, self._semantic_pcc(groups, scores))
        assessed.append({"groups": groups, "scores": scores,
                         "resolution": min(row["resolution"] for row in partitions),
                         "seed": min(row["seed"] for row in partitions),
                         "quality": quality, "source": "graph_refinement"})
        print(f"   L{layer} 图边补查：候选边={len(pairs)}，"
              f"同层合并={len(selected)}，ΔH={quality['delta_h']:.3f}，"
              f"J={quality['J']:.3f}", flush=True)

    def _resolve_upper_names(self, groups, names, examples, scores, decisions,
                             cache=None):
        resolved = []
        occupied = {name for _, name, _ in groups}
        for children, name, pass_through in groups:
            duplicate = next((index for index, row in enumerate(resolved)
                              if row[1] == name), None)
            if duplicate is None:
                resolved.append((children, name, pass_through))
                continue
            other_children, _, other_pass = resolved[duplicate]
            first = [{"name": names[i], "representative_terms": examples[i]}
                     for i in other_children]
            second = [{"name": names[i], "representative_terms": examples[i]}
                      for i in children]
            collision_key = (tuple(other_children), tuple(children), name,
                             tuple(sorted(occupied)))
            if cache is not None and collision_key in cache.collisions:
                decision = cache.collisions[collision_key]
            else:
                same, renamed, reason = self.llm.resolve_upper_collision(
                    name, second if pass_through and not other_pass else first,
                    first if pass_through and not other_pass else second, occupied)
                decision = {"same": same, "renamed": renamed, "reason": reason}
                if cache is not None:
                    cache.record_collision(other_children, children, name,
                                           occupied, decision)
            same, renamed, reason = (decision["same"], decision["renamed"],
                                     decision["reason"])
            decisions.append({"candidate_source": "upper_name_collision",
                              "child_ids": children, "original_name": name,
                              "resolved_name": renamed, "merged": same,
                              "reason": reason})
            if same:
                if pass_through:
                    child_id = children[0]
                    edge_key = (child_id, name)
                    if cache is not None and edge_key in cache.collision_edges:
                        scores[child_id] = cache.collision_edges[edge_key]
                    else:
                        scores[child_id] = self.llm.score_existing_parent(
                            second[0], name)
                        if cache is not None:
                            cache.record_collision_edge(child_id, name,
                                                        scores[child_id])
                if other_pass:
                    child_id = other_children[0]
                    edge_key = (child_id, name)
                    if cache is not None and edge_key in cache.collision_edges:
                        scores[child_id] = cache.collision_edges[edge_key]
                    else:
                        scores[child_id] = self.llm.score_existing_parent(
                            first[0], name)
                        if cache is not None:
                            cache.record_collision_edge(child_id, name,
                                                        scores[child_id])
                if any(scores.get(i, 0) < 1 for i in children + other_children):
                    raise RuntimeError(f"同名合并 {name} 存在不被父概念涵盖的子边")
                resolved[duplicate] = (sorted(other_children + children), name, False)
            else:
                occupied.add(renamed)
                if pass_through and not other_pass:
                    resolved[duplicate] = (other_children, renamed, False)
                    resolved.append((children, name, True))
                else:
                    resolved.append((children, renamed, pass_through))
        return sorted(resolved, key=lambda row: min(row[0]))


    @staticmethod
    def _semantic_pcc(groups, score_by_child):
        parent_scores = []
        for children, _, _ in groups:
            if len(children) >= 2:
                parent_scores.append(np.mean([
                    score_by_child[child] for child in children]) / 3.0)
        return float(np.mean(parent_scores)) if parent_scores else 0.0

    def _preflight_layer_groups(self, groups, X, previous_leaves, layer,
                                score_by_child=None):
        """Check semantic coverage and SC before committing a candidate layer."""
        labels = np.empty(len(X), dtype=int)
        for cid, (children, _, _) in enumerate(groups):
            labels[children] = cid
        pcc = self._semantic_pcc(groups, score_by_child or {})
        sc = (float(silhouette_score(X, labels, metric="cosine"))
              if 1 < len(groups) < len(X) else 0.0)
        return pcc, sc

    def _build_l1_from_initial_labels(self):
        """
        使用 initial_labels 直接构建 L1 细粒度叶簇层。

        层级定义：
        - L0：原始样本 / 原始术语；
        - L1：由 Stage3 最佳聚类结果得到的细粒度叶簇；
        - L2/L3/L4...：在 L1 簇代表中心基础上继续做层次聚类、LLM命名、ACRL代表学习和HCR-Stop停止判断。

        注意：
        - initial_labels 应传入最佳聚类结果 pred_labels_cosine；
        - reference_labels 只用于 DP 评价，不参与构树；
        - L1 会调用 LLM 生成概念名；同名仅在确认属于同一细类后合并。
        """
        print(f"\n{'=' * 70}")
        print("【第1层】使用 Stage3 最佳聚类标签构建细粒度叶簇 L1")
        print(f"{'=' * 70}")

        labels = np.asarray(self.labels)
        unique_labels = sorted(np.unique(labels).tolist(), key=lambda x: str(x))

        concepts = {}
        clusters_names = {}
        member_indices = {}
        leaf_indices = {}
        cluster_centers = {}
        center_weights = {}
        clusters_data = {}
        original_predicted_labels = {}
        source_term_counts = {}

        for new_cid, old_label in enumerate(unique_labels):
            idxs = np.where(labels == old_label)[0].tolist()
            original_predicted_labels[new_cid] = str(old_label)
            source_term_counts[new_cid] = len(idxs)

            clusters_names[new_cid] = [str(self.entities[i]) for i in idxs]
            member_indices[new_cid] = idxs
            leaf_indices[new_cid] = idxs

            adaptive_center, weights = self.acrl.compute_adaptive_center(
                self.embeddings[idxs],
                cluster_id=new_cid,
                layer=1
            )

            cluster_centers[new_cid] = adaptive_center
            center_weights[new_cid] = weights.tolist()
            representative_positions = np.argsort(-weights, kind="stable")[:8]
            clusters_data[new_cid] = [str(self.entities[idxs[pos]])
                                      for pos in representative_positions]

        print(f"   L1叶簇构建完成")
        print(f"   L1簇数: {len(unique_labels)}")
        print(f"   平均簇大小: {len(self.entities) / max(len(unique_labels), 1):.2f}")

        if not self.l1_category_names:
            # 为当前数据集提供 L1 候选类别编号与名称清单。
            raise ValueError("L1 受限命名需要当前数据集的类别编号与名称清单")
        signature = hashlib.sha256(json.dumps(
            {"entities": self.entities, "labels": labels.tolist(),
             "catalog": self.l1_category_names, "model": self.llm.model_name,
             "prompt_profile": _prompt_profile_signature()},
            ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        choice_path = os.path.join(self.output_dir, "l1_name_choices.json")
        choices = {}
        if os.path.isfile(choice_path):
            with open(choice_path, encoding="utf-8") as stream:
                saved = json.load(stream)
            if saved.get("signature") != signature:
                raise ValueError("L1 命名缓存与当前术语、簇或类别清单不匹配")
            choices = {int(cid): row for cid, row in saved["choices"].items()}
        for cid, terms in tqdm(clusters_names.items(), desc="L1 受限类别选择", ncols=80):
            if cid in choices:
                continue
            choices[cid] = self.llm.choose_l1_category(
                cid, terms, clusters_data[cid], self.l1_category_names)
            temporary = choice_path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump({"signature": signature, "choices": choices}, stream,
                          ensure_ascii=False, indent=2)
            os.replace(temporary, choice_path)
        if set(choices) != set(clusters_names):
            raise RuntimeError("L1 未完成全部预测簇的目录选择")
        verified_path = os.path.join(self.output_dir, "l1_name_choices_verified.json")
        verified = {}
        if os.path.isfile(verified_path):
            with open(verified_path, encoding="utf-8") as stream:
                saved = json.load(stream)
            if saved.get("signature") != signature or saved.get("version") != 2:
                raise ValueError("L1 类别复核缓存与当前术语、簇或清单不匹配")
            verified = {int(cid): row for cid, row in saved["choices"].items()}
            if any(cid not in choices or row.get("initial_label") != choices[cid]["label"]
                   for cid, row in verified.items()):
                raise ValueError("L1 类别复核缓存与首次选择不一致")
        for cid, terms in tqdm(clusters_names.items(), desc="L1 类别复核", ncols=80):
            if cid in verified:
                continue
            verified[cid] = self.llm.verify_l1_category(
                cid, terms, clusters_data[cid], self.l1_category_names, choices[cid])
            temporary = verified_path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump({"signature": signature, "version": 2, "choices": verified},
                          stream, ensure_ascii=False, indent=2)
            os.replace(temporary, verified_path)
        if set(verified) != set(clusters_names):
            raise RuntimeError("L1 类别复核未覆盖全部预测簇")
        choices = verified
        by_label = {}
        for cid, choice in choices.items():
            by_label.setdefault(choice["label"], []).append(cid)
        source_leaves = leaf_indices
        l1_name_audits = {cid: choice for cid, choice in choices.items()}
        resolved = [(sorted(sources), self.l1_category_names[label], label)
                    for label, sources in sorted(by_label.items(),
                                                 key=lambda item: min(item[1]))]
        collision_decisions = [
            {"label": label, "name": name, "source_ids": sources}
            for sources, name, label in resolved if len(sources) > 1]
        concepts, clusters_names, member_indices = {}, {}, {}
        leaf_indices, cluster_centers, center_weights, clusters_data = {}, {}, {}, {}
        source_cluster_ids, category_label_ids = {}, {}
        self.llm.provisional_l1_names = {}
        for cid, (source_ids, name, label) in enumerate(resolved):
            indices = sorted(i for source in source_ids for i in source_leaves[source])
            if len(indices) != len(set(indices)):
                raise ValueError("L1 同名合并发现重复术语")
            center, weights = self.acrl.compute_adaptive_center(
                self.embeddings[indices], cluster_id=cid, layer=1)
            concepts[cid] = name
            clusters_names[cid] = [str(self.entities[i]) for i in indices]
            member_indices[cid] = indices
            leaf_indices[cid] = indices
            cluster_centers[cid] = center
            center_weights[cid] = weights.tolist()
            source_cluster_ids[cid] = source_ids
            category_label_ids[cid] = label
            positions = np.argsort(-weights, kind="stable")[:8]
            clusters_data[cid] = [str(self.entities[indices[pos]]) for pos in positions]
            uncertain = [source for source in source_ids
                         if not choices[source]["coherent"] or
                         choices[source]["category_fit"] < 2]
            if uncertain:
                self.llm.provisional_l1_names[cid] = {
                    "name": name, "reason": "原预测簇内部一致性需复核：" +
                    "、".join(str(source) for source in uncertain)}
        if sorted(i for indices in leaf_indices.values() for i in indices) != list(
                range(len(self.entities))):
            raise ValueError("L1 同名消解后未完整且唯一覆盖原始术语")
        print(f"   L1 清单同名合并：{len(unique_labels)} 个原始细簇 → {len(concepts)} 个节点；"
              f"同名组 {len(collision_decisions)} 组")

        preview_num = min(20, len(concepts))
        print(f"   L1概念预览（前{preview_num}个）:")
        for cid in list(concepts.keys())[:preview_num]:
            print(f"      L1-Cluster{cid}: {concepts[cid]} ({len(clusters_names[cid])}项)")
        if len(concepts) > preview_num:
            print(f"      ... 其余 {len(concepts) - preview_num} 个 L1 细粒度簇已省略打印")

        self.ontology["L1"] = {
            "concepts": concepts,
            "clusters": clusters_names,
            "centers_index": list(concepts.keys()),
            # L1 的直接子节点是 L0 样本节点，因此 member_indices 与 leaf_indices 在 L1 层一致。
            "member_indices": member_indices,
            "leaf_indices": leaf_indices,
            "n_clusters": len(concepts),
            "acrl_weights": {int(k): v for k, v in center_weights.items()},
            "representative_terms": clusters_data,
            "source_cluster_ids": source_cluster_ids,
            "source_predicted_labels": original_predicted_labels,
            "source_term_counts": source_term_counts,
            "category_label_ids": category_label_ids,
            "l1_name_audit": l1_name_audits,
            "name_collision_decisions": collision_decisions,
            "has_provisional_l1_descendant": {
                cid: cid in self.llm.provisional_l1_names for cid in concepts},
            "child_layer": "L0",
            "parent_layer": "L2",
            "description": "由Stage3最佳聚类标签直接生成的细粒度叶簇层"
        }

        self.leaf_clusters_by_layer["L1"] = leaf_indices

        X_l1 = np.vstack([cluster_centers[cid] for cid in concepts.keys()])
        names_l1 = [str(concepts[cid]) for cid in concepts.keys()]
        prev_leaf_indices = {
            local_idx: leaf_indices[cid]
            for local_idx, cid in enumerate(concepts.keys())
        }

        return X_l1, names_l1, prev_leaf_indices

    def _sorted_generated_layers(self) -> List[str]:
        """返回除 L0 外的生成层名称。"""
        return sorted(
            [k for k in self.ontology.keys() if k.startswith("L") and k != "L0"],
            key=lambda x: int(x[1:])
        )

    def compute_dendrogram_purity(self, reference_labels: Optional[np.ndarray] = None) -> float:
        """计算 Dendrogram Purity。"""
        labels = np.asarray(reference_labels if reference_labels is not None else self.reference_labels)
        if len(labels) != len(self.entities):
            raise ValueError("reference_labels长度必须与实体数量一致")
        layer_names = self._sorted_generated_layers()
        if not layer_names:
            return 0.0
        leaf_to_cluster_by_layer = []
        cluster_leaves_by_layer = []
        for layer_name in layer_names:
            layer_data = self.ontology[layer_name]
            leaf_indices_map = layer_data.get("leaf_indices", {})
            leaf_to_cluster = {}
            normalized_clusters = {}
            for cid, leaf_indices in leaf_indices_map.items():
                idxs = [int(i) for i in leaf_indices if 0 <= int(i) < len(labels)]
                normalized_clusters[cid] = idxs
                for idx in idxs:
                    leaf_to_cluster[idx] = cid
            leaf_to_cluster_by_layer.append(leaf_to_cluster)
            cluster_leaves_by_layer.append(normalized_clusters)
        total_score = 0.0
        pair_count = 0
        all_indices = np.arange(len(labels))
        for cls in np.unique(labels):
            cls_indices = all_indices[labels == cls]
            if len(cls_indices) < 2:
                continue
            cls_set = set(int(i) for i in cls_indices)
            for pos_i in range(len(cls_indices)):
                i = int(cls_indices[pos_i])
                for pos_j in range(pos_i + 1, len(cls_indices)):
                    j = int(cls_indices[pos_j])
                    ancestor_leaves = None
                    for leaf_to_cluster, cluster_leaves in zip(leaf_to_cluster_by_layer, cluster_leaves_by_layer):
                        ci = leaf_to_cluster.get(i)
                        cj = leaf_to_cluster.get(j)
                        if ci is not None and ci == cj:
                            ancestor_leaves = cluster_leaves[ci]
                            break
                    if ancestor_leaves is None:
                        ancestor_leaves = list(range(len(labels)))
                    same_cls_count = sum(1 for idx in ancestor_leaves if idx in cls_set)
                    total_score += same_cls_count / max(len(ancestor_leaves), 1)
                    pair_count += 1
        return float(total_score / pair_count) if pair_count > 0 else 0.0

    def _count_concept_layer_nodes(self, layer_name: str) -> int:
        """统计某一概念层的节点数。L0为实例/术语层，不作为概念层。"""
        if layer_name not in self.ontology or layer_name == "L0":
            return 0
        layer_data = self.ontology[layer_name]
        concepts = layer_data.get("concepts", {})
        if concepts:
            return int(len(concepts) - len(layer_data.get("pass_through_nodes", [])))
        return int(layer_data.get("n_clusters", 0))

    def _concept_parent_child_edge_stats(self) -> tuple:
        """
        统计概念层之间的父子继承边。

        口径：
        - L0 是原始样本/术语实例层，不计入本体概念节点；
        - 只统计 L2→L1、L3→L2、...、Lk→L(k-1) 的概念父子边；
        - 不统计 L1→L0 的样本归属边，避免样本数量导致 IR 虚高。

        Returns:
            concept_edges: 概念层父子边数量
            parent_nodes_with_children: 至少拥有一个子概念的父概念节点数量
            concept_nodes: L1 到 Lk 的概念节点总数
        """
        layer_names = self._sorted_generated_layers()
        concept_nodes = sum(self._count_concept_layer_nodes(layer) for layer in layer_names)

        concept_edges = 0
        parent_nodes_with_children = 0

        for layer_name in layer_names:
            if layer_name == "L1":
                continue
            layer_data = self.ontology.get(layer_name, {})
            child_layer = layer_data.get("child_layer")

            # 只统计概念层之间的边，不统计指向L0样本层的边。
            if child_layer is None or child_layer == "L0":
                continue

            member_indices = layer_data.get("member_indices", {})
            pass_through = set(layer_data.get("pass_through_nodes", []))
            for cid, children in member_indices.items():
                if int(cid) in pass_through:
                    continue
                if children is None:
                    continue
                # member_indices 在 L2+ 中表示当前父概念包含的上一层子概念ID。
                child_ids = [int(x) for x in children]
                n_children = len(child_ids)
                if n_children > 0:
                    concept_edges += n_children
                    parent_nodes_with_children += 1

        return int(concept_edges), int(parent_nodes_with_children), int(concept_nodes)

    def compute_inheritance_richness(self) -> float:
        """
        计算 Inheritance Richness (IR)。

        参考 OntoQA 的继承丰富度定义，IR 表示平均每个类具有多少个子类。
        在本文的层次树中，L1及以上视为概念节点，L0视为实例/术语节点，
        因此只计算概念层之间的父子继承边。

        IR = 概念层父子边数量 / 概念节点总数
        """
        concept_edges, _, concept_nodes = self._concept_parent_child_edge_stats()
        if concept_nodes <= 0:
            return 0.0
        return float(concept_edges / concept_nodes)

    def compute_average_fanout(self) -> float:
        """
        计算 Average Fan-out（平均扇出度）。

        Average Fan-out = 概念层父子边数量 / 有子概念的父概念节点数量
        """
        concept_edges, parent_nodes_with_children, _ = self._concept_parent_child_edge_stats()
        if parent_nodes_with_children <= 0:
            return 0.0
        return float(concept_edges / parent_nodes_with_children)

    def compute_hierarchy_quality_metrics(self, reference_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """计算本体树评价指标：Depth、Dendrogram Purity、IR 和 Average Fan-out。"""
        labels = np.asarray(reference_labels if reference_labels is not None else self.reference_labels)
        layer_names = self._sorted_generated_layers()
        final_layer = layer_names[-1] if layer_names else None

        concept_edges, parent_nodes_with_children, concept_nodes = self._concept_parent_child_edge_stats()
        metrics = {
            "hierarchy_depth": int(len(self.ontology)),
            "concept_depth": int(len(layer_names)),
            "dendrogram_purity": float(self.compute_dendrogram_purity(labels)),
            "inheritance_richness": float(self.compute_inheritance_richness()),
            "average_fanout": float(self.compute_average_fanout()),
            "concept_edge_count": int(concept_edges),
            "concept_node_count": int(concept_nodes),
            "parent_concept_count": int(parent_nodes_with_children),
            "final_layer": final_layer,
            "final_concept_count": int(self._count_concept_layer_nodes(final_layer)) if final_layer else 0
        }
        self.hierarchy_metrics = metrics
        return metrics

    def run(self):
        """
        主流程：从术语语义向量开始，逐层执行语义聚类、ACRL簇代表学习、
        LLM概念抽象和HCR-Stop判别。
        """
        print(f"\n{'=' * 70}")
        print("开始构建层次化知识体系（HCRL）")
        print(f"{'=' * 70}")

        # 第0层：原始术语层，作为第一轮语义聚类输入
        print(f"\n{'=' * 70}")
        print(f"【第0层】初始化 - 原始运维术语向量")
        print(f"{'=' * 70}")

        self.prev_centers = self.embeddings

        # L0 是完整层次树的样本节点层：每个原始术语都是一个独立节点。
        # 不能把 L0 写成一个“原始运维术语”总节点，否则 L0→L1→L2... 的树结构不严谨。
        self.ontology["L0"] = {
            "concepts": {
                int(i): str(entity)
                for i, entity in enumerate(self.entities)
            },
            "clusters": {
                int(i): [str(entity)]
                for i, entity in enumerate(self.entities)
            },
            "centers_index": list(range(len(self.entities))),
            # L0 是最底层样本节点，没有更低一级直接子节点。
            "member_indices": {
                int(i): []
                for i in range(len(self.entities))
            },
            # 每个 L0 节点覆盖它自身这个原始样本。
            "leaf_indices": {
                int(i): [int(i)]
                for i in range(len(self.entities))
            },
            "n_clusters": len(self.entities),
            "child_layer": None,
            "parent_layer": "L1",
            "description": "原始样本节点层，每个节点对应一个原始运维术语"
        }
        self.leaf_clusters_by_layer["L0"] = {
            int(i): [int(i)]
            for i in range(len(self.entities))
        }
        print(f"   原始术语数: {len(self.entities)}")
        print(f"   L0样本节点数: {self.ontology['L0']['n_clusters']}")
        print(f"   向量维度: {self.embeddings.shape[1] if self.embeddings.ndim == 2 else 'unknown'}")

        # 第1层：直接使用 Stage3 最佳聚类标签构建细粒度叶簇 L1
        # 第2层及以上：在 L1 簇代表中心上继续抽象聚类 + LLM命名 + HCR-Stop
        X, names, prev_leaf_indices = self._build_l1_from_initial_labels()
        self.leiden_reference_count = len(X)
        node_examples = [self.ontology["L1"]["representative_terms"][i]
                         for i in range(len(X))]
        provisional_flags = [i in self.llm.provisional_l1_names for i in range(len(X))]

        self.prev_centers = X
        l1_sizes = np.array([len(prev_leaf_indices[i]) for i in range(len(X))], dtype=float)
        l1_probabilities = l1_sizes / l1_sizes.sum()
        self.hcr_stop.prev_entropy = float(-np.sum(
            l1_probabilities * np.log(l1_probabilities + 1e-10)))
        layer = 2
        last_completed_layer = 1
        while layer <= self.max_layers:
            tag = f"L{layer}"
            print(f"\n{'=' * 70}")
            print(f"【第{layer}层】语义聚类 + ACRL簇代表学习 (抽象度级别: {self.llm._get_abstract_level(layer)})")
            print(f"{'=' * 70}")

            if len(X) < 2:
                print(f"   当前输入仅 {len(X)} 个中心，停止向上扩展")
                self.termination_reason = "输入概念数不足，无法继续形成有效父簇"
                break

            self._current_leaf_indices = prev_leaf_indices
            reviewed_groups, merge_decisions, search_info = self._plan_leiden_groups(
                X, names, node_examples, provisional_flags, layer)
            rollbacks = []
            self.layer_linkages[tag] = {"candidate_search": search_info,
                                        "semantic_review": merge_decisions,
                                        "individual_rollbacks": rollbacks,
                                        "merge_mode": "is_a"}
            if not any(len(group) > 1 for group, _, _ in reviewed_groups):
                self.termination_reason = (
                    "充分搜索后没有可命名的有效合并，保留当前层")
                print(f"   {self.termination_reason}，保留 L{last_completed_layer} 为顶层")
                break
            if self.hcr_stop.abstraction_stalled(search_info["selected_quality"]):
                self.termination_reason = "连续候选层抽象收益过低，保留上一有效层"
                print(f"   {self.termination_reason}，保留 L{last_completed_layer} 为顶层")
                break
            if self.hcr_stop.candidate_quality_declines(
                    search_info["selected_quality"]):
                self.quality_decline_count += 1
            else:
                self.quality_decline_count = 0
            if self.quality_decline_count >= self.hcr_stop.patience:
                self.termination_reason = "连续候选层的 SC 与 PCC 均明显下降"
                print(f"   {self.termination_reason}，保留 L{last_completed_layer} 为顶层")
                break
            print(f"   Leiden 图边={search_info.get('edge_count', 0)}，候选组="
                  f"{search_info['candidate_count']}，命名审核="
                  f"{search_info['llm_review_count']}，合并="
                  f"{sum(len(group) > 1 for group, _, _ in reviewed_groups)}，"
                  f"单项回退={len(rollbacks)}")

            clusters_names, member_indices, concepts, cluster_centers = {}, {}, {}, {}
            cut_threshold = None
            labels = np.empty(len(X), dtype=int)
            accepted_names = {}
            pass_through_nodes = []
            for cid, (child_ids, parent_name, is_pass_through) in enumerate(reviewed_groups):
                labels[child_ids] = cid
                accepted_names[cid] = parent_name
                if is_pass_through:
                    pass_through_nodes.append(cid)
            propagated_flags = {cid: any(provisional_flags[i] for i in child_ids)
                                for cid, (child_ids, _, _) in enumerate(reviewed_groups)}
            unique_cids = list(range(len(reviewed_groups)))
            self.layer_linkages[tag]["accepted_cluster_count"] = len(unique_cids)
            self.layer_linkages[tag]["accepted_merge_count"] = sum(
                len(group) > 1 for group, _, _ in reviewed_groups)

            clusters_data = {}
            center_weights = {}
            leaf_indices = {}
            representative_terms = {}
            for cid in unique_cids:
                idxs = np.where(labels == cid)[0].tolist()
                samples = [names[i] for i in idxs]
                clusters_names[cid] = samples
                member_indices[cid] = idxs
                leaves = []
                for idx in idxs:
                    leaves.extend(prev_leaf_indices[idx])
                leaf_indices[cid] = sorted(set(int(x) for x in leaves))

                adaptive_center, weights = self.acrl.compute_adaptive_center(
                    X[idxs], cluster_id=cid, layer=layer)
                cluster_centers[cid] = adaptive_center
                center_weights[cid] = weights.tolist()
                child_vectors = X[idxs]
                denominator = np.linalg.norm(child_vectors, axis=1) * np.linalg.norm(adaptive_center)
                similarities = np.divide(
                    child_vectors @ adaptive_center, denominator,
                    out=np.zeros(len(idxs), dtype=float), where=denominator > 1e-12)
                representative_positions = np.argsort(-similarities, kind="stable")[:8]
                clusters_data[cid] = [samples[pos] for pos in representative_positions]
                representative_terms[cid] = (
                    [node_examples[i][0] for i in idxs if node_examples[i]] +
                    [term for i in idxs for term in node_examples[i][1:2]]
                )[:8]

            all_leaves = [leaf for cid in unique_cids for leaf in leaf_indices[cid]]
            if sorted(all_leaves) != list(range(len(self.entities))):
                raise ValueError(f"{tag} 的父簇未完整且唯一地覆盖原始术语")

            current_centers_array = np.vstack([cluster_centers[cid] for cid in unique_cids])
            parent_to_children = {
                local_id: member_indices[cid]
                for local_id, cid in enumerate(unique_cids)
            }
            acrl_pcc = self.hcr_stop.compute_parent_child_consistency(
                parent_centers=current_centers_array,
                child_centers=X,
                parent_to_children=parent_to_children)
            pcc = self._semantic_pcc(
                reviewed_groups, search_info.get("semantic_scores_by_child", {}))
            self.layer_linkages[tag]["acrl_parent_child_cosine"] = acrl_pcc
            stop, reason, _, J = self.hcr_stop.adaptive_stopping_criterion(
                X=X,
                labels=labels,
                layer=layer,
                pcc=pcc,
                input_count=len(X),
                leaf_counts=np.array([len(prev_leaf_indices[i]) for i in range(len(X))]),
                enforce_pcc_floor=False
            )

            concepts = accepted_names

            if set(concepts) != set(unique_cids):
                raise ValueError(f"{tag} 概念命名结果与聚类编号不一致")
            for cid in unique_cids:
                print(f"   {tag}-Cluster{cid}: {concepts[cid]} ({len(clusters_names[cid])}项)")

            self.ontology[tag] = {
                "concepts": concepts,
                "clusters": clusters_names,
                "centers_index": unique_cids,
                # L2及以上：member_indices 表示当前节点直接包含的上一层子节点编号。
                "member_indices": member_indices,
                # leaf_indices 表示当前节点最终覆盖的 L0 原始样本编号。
                "leaf_indices": leaf_indices,
                "n_clusters": len(concepts),
                "acrl_weights": {int(k): center_weights.get(k, []) for k in center_weights},
                "pass_through_nodes": pass_through_nodes,
                "merge_mode": "is_a",
                "relation_type": "is_a",
                "semantic_review": merge_decisions,
                "representative_terms": representative_terms,
                "has_provisional_l1_descendant": propagated_flags,
                "auto_cut_threshold": cut_threshold,
                "linkage": self.layer_linkages.get(tag, {}),
                "child_layer": f"L{layer - 1}",
                "parent_layer": f"L{layer + 1}" if layer < self.max_layers else None
            }
            self.leaf_clusters_by_layer[tag] = leaf_indices
            last_completed_layer = layer
            if stop:
                self.ontology[tag]["parent_layer"] = None
                self.termination_reason = reason
                print(f"\n{reason}")
                break

            self.prev_centers = np.vstack([cluster_centers[cid] for cid in unique_cids])
            X = self.prev_centers
            names = [concepts[cid] for cid in unique_cids]
            node_examples = [representative_terms[cid] for cid in unique_cids]
            provisional_flags = [propagated_flags[cid] for cid in unique_cids]
            prev_leaf_indices = {local_idx: leaf_indices[cid] for local_idx, cid in enumerate(unique_cids)}
            layer += 1

        if self.termination_reason is None:
            self.termination_reason = f"达到 max_layers={self.max_layers} 上限"
        self.ontology[f"L{last_completed_layer}"]["parent_layer"] = None
        # 单棵树只记录结构规模。DP 由实验脚本用人工参考类别计算；
        # ECS 需要同组不同 seed 的完整树，不能在这里给出单树分数。
        self.hierarchy_metrics = {
            "concept_depth": last_completed_layer,
            "final_layer": f"L{last_completed_layer}",
            "top_level_node_count": self.ontology[f"L{last_completed_layer}"]["n_clusters"],
            "nodes_by_layer": {
                layer: self.ontology[f"L{layer}"]["n_clusters"]
                for layer in range(1, last_completed_layer + 1)
            },
        }
        print("\n选定的层次评价指标:")
        print(f"   概念深度（不计 L0）: {last_completed_layer}")
        print("   DP: 由运行脚本使用人工参考类别计算")
        print("   ECS: 需同组至少两个 seed 的完整树，跨 seed 汇总时计算")
        self._save_results()
        print(f"\n{'=' * 70}")
        print(f"层次化知识构建完成，概念深度 {last_completed_layer}（另有 L0 术语层）")
        print(f"{'=' * 70}")

        return self.ontology

    def _save_results(self):
        """保存结果"""
        out_path = os.path.join(self.output_dir, "ontology_structure.json")
        out = {
            "timestamp": datetime.now().isoformat(),
            "ontology": self.ontology,
            "candidate_reviews": {
                layer: {key: value for key, value in record.items()
                        if key in {"candidate_search", "semantic_review",
                                   "individual_rollbacks", "accepted_cluster_count",
                                   "accepted_merge_count"}}
                for layer, record in self.layer_linkages.items()
            },
            "summary": {
                "total_layers": len(self.ontology),
                "termination_reason": self.termination_reason,
                "hierarchy_metrics": self.hierarchy_metrics,
                "layer_stats": {
                    layer: {
                        "n_clusters": data["n_clusters"],
                        "concepts": list(data["concepts"].values())
                    }
                    for layer, data in self.ontology.items()
                }
            }
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Ontology结构已保存到 {out_path}")

        history_path = os.path.join(self.output_dir, "metrics_history.json")
        self.hcr_stop.export_history(history_path)

        hierarchy_metrics_path = os.path.join(self.output_dir, "hierarchy_quality_metrics.json")
        with open(hierarchy_metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.hierarchy_metrics, f, ensure_ascii=False, indent=2)
        print(f"层次结构统计已保存到 {hierarchy_metrics_path}")

        tree_path = os.path.join(self.output_dir, "hierarchy_tree.txt")
        with open(tree_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("层次化知识体系\n")
            f.write("=" * 70 + "\n\n")

            for layer_name in sorted(self.ontology.keys()):
                layer_data = self.ontology[layer_name]
                f.write(f"\n【{layer_name}】- {layer_data['n_clusters']} 个概念\n")
                f.write("-" * 70 + "\n")

                for cid, concept in layer_data['concepts'].items():
                    cluster_size = len(layer_data['clusters'][cid])
                    f.write(f"  {concept} ({cluster_size}项)\n")

                    if layer_name == "L0" and cluster_size <= 10:
                        for sample in layer_data['clusters'][cid][:10]:
                            f.write(f"    - {sample}\n")
                    elif layer_name != "L0" and cluster_size <= 10:
                        for sub_concept in layer_data['clusters'][cid]:
                            f.write(f"    - {sub_concept}\n")

                f.write("\n")

        print(f"层次树文本已保存到 {tree_path}")


def run_hcr_stop_ablation(
        embeddings: np.ndarray,
        entities: List[str],
        reference_labels: np.ndarray,
        initial_labels: Optional[np.ndarray] = None,
        output_dir: str = "./hcr_stop_ablation",
        fixed_max_layers: int = 14,
        hcr_max_layers: int = 14,
        llm_api_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model_name: str = "qwen-max",
        use_parallel_llm: bool = False,
        acrl_alpha: float = 0.6,
        min_clusters_per_layer: int = 2,
        min_aggregation_coverage: float = 0.0,
        hcr_min_layers: int = 0,
        hcr_patience: int = 1
) -> Dict[str, Dict[str, float]]:
    """
    HCR-Stop 消融实验。

    本版实验设计：
    1. 固定深度上限扫描：Depth 上限 = 5 ~ 15；语义审核可提前停止。
       注意：代码中的 max_layers 不包含 L0，所以 Depth=d 对应 max_layers=d-1。
    2. HCR-Stop：最大允许到 Depth=15，但由 HCR-Stop 自动决定实际停止层。
    3. 输出指标：Depth, Dendrogram Purity, Inheritance Richness, Average Fan-out。
    """
    os.makedirs(output_dir, exist_ok=True)

    if initial_labels is None:
        initial_labels = reference_labels

    initial_labels = np.asarray(initial_labels)
    reference_labels = np.asarray(reference_labels)

    if len(initial_labels) != len(entities):
        raise ValueError("initial_labels长度必须与entities数量一致")
    if len(reference_labels) != len(entities):
        raise ValueError("reference_labels长度必须与entities数量一致")

    # 1. 构造固定深度扫描配置
    # fixed_max_layers=14 对应最终 Depth=15
    # 因此这里扫描 Depth=5~15，即 max_layers=4~14
    fixed_depths = list(range(5, fixed_max_layers + 2))

    configs = {}

    for depth in fixed_depths:
        configs[f"Fixed-depth-{depth}"] = {
            "use_hcr_stop": False,
            "use_pcc": False,
            "max_layers": depth - 1,
            "target_depth": depth,
            "description": f"允许的最大层次深度为 Depth={depth}"
        }

    # HCR-Stop 最大允许到 Depth=hcr_max_layers+1
    configs["HCR-Stop"] = {
        "use_hcr_stop": True,
        "use_pcc": True,
        "max_layers": hcr_max_layers,
        "target_depth": hcr_max_layers + 1,
        "description": f"HCR-Stop自动停止，最大允许Depth={hcr_max_layers + 1}"
    }

    results = {}

    # 2. 逐个配置运行
    for method_name, cfg in configs.items():
        print(f"\n{'=' * 70}")
        print(f"运行层次深度扫描 / HCR-Stop实验: {method_name}")
        print(f"说明: {cfg['description']}")
        print(f"max_layers={cfg['max_layers']}，理论最大Depth={cfg['max_layers'] + 1}")
        print(f"{'=' * 70}")

        safe_name = (
            method_name
            .replace("/", "without")
            .replace(" ", "_")
            .replace("=", "_")
            .replace(":", "_")
        )

        method_dir = os.path.join(output_dir, safe_name)

        builder = HierarchicalKnowledgeBuilder(
            initial_embeddings=embeddings,
            initial_entities=entities,
            initial_labels=initial_labels,
            reference_labels=reference_labels,
            output_dir=method_dir,
            max_layers=cfg["max_layers"],
            llm_api_url=llm_api_url,
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
            use_parallel_llm=use_parallel_llm,
            acrl_alpha=acrl_alpha,
            use_hcr_stop=cfg["use_hcr_stop"],
            use_pcc=cfg["use_pcc"],
            min_clusters_per_layer=min_clusters_per_layer,
            min_aggregation_coverage=min_aggregation_coverage,
            hcr_min_layers=hcr_min_layers,
            hcr_patience=hcr_patience
        )

        ontology = builder.run()
        metrics = builder.compute_hierarchy_quality_metrics(reference_labels)

        # 真实终止原因可能是候选无效或达到上限，不一定是最后一次 HCR 判定。
        final_stop_reason = builder.termination_reason or ""

        results[method_name] = {
            "target_depth": int(cfg["target_depth"]),
            "hierarchy_depth": int(metrics.get("hierarchy_depth", len(ontology))),
            "concept_depth": int(metrics.get("concept_depth", max(len(ontology) - 1, 0))),
            "dendrogram_purity": float(metrics["dendrogram_purity"]),
            "inheritance_richness": float(metrics.get("inheritance_richness", 0.0)),
            "average_fanout": float(metrics.get("average_fanout", 0.0)),
            "final_layer": metrics.get("final_layer"),
            "final_concept_count": int(metrics.get("final_concept_count", 0)),
            "concept_edge_count": int(metrics.get("concept_edge_count", 0)),
            "concept_node_count": int(metrics.get("concept_node_count", 0)),
            "parent_concept_count": int(metrics.get("parent_concept_count", 0)),
            "use_hcr_stop": bool(cfg["use_hcr_stop"]),
            "use_pcc": bool(cfg["use_pcc"]),
            "max_layers": int(cfg["max_layers"]),
            "min_aggregation_coverage": float(min_aggregation_coverage),
            "description": cfg["description"],
            "stop_reason": final_stop_reason
        }

    # 3. 保存 JSON
    summary_path = os.path.join(output_dir, "hcr_stop_ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 4. 保存论文表格：固定深度扫描 + HCR-Stop
    table_path = os.path.join(output_dir, "table_3_4_hcr_stop_ablation.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("| 方法设置 | Target Depth | Actual Depth | Dendrogram Purity | Inheritance Richness | Average Fan-out | Stop Reason |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")

        for method_name, row in results.items():
            f.write(
                f"| {method_name} | "
                f"{row['target_depth']} | "
                f"{row['hierarchy_depth']} | "
                f"{row['dendrogram_purity']:.4f} | "
                f"{row.get('inheritance_richness', 0.0):.4f} | "
                f"{row.get('average_fanout', 0.0):.4f} | "
                f"{row.get('stop_reason', '')} |\n"
            )

    # 5. 额外保存一个“收益变化表”：相邻固定深度的指标增量
    delta_table_path = os.path.join(output_dir, "table_fixed_depth_delta.md")

    fixed_rows = [
        (name, row)
        for name, row in results.items()
        if name.startswith("Fixed-depth-")
    ]
    fixed_rows = sorted(fixed_rows, key=lambda x: x[1]["hierarchy_depth"])

    with open(delta_table_path, "w", encoding="utf-8") as f:
        f.write("| Depth变化 | ΔDP | ΔIR | ΔAverage Fan-out |\n")
        f.write("|---|---:|---:|---:|\n")

        for i in range(1, len(fixed_rows)):
            prev_name, prev_row = fixed_rows[i - 1]
            curr_name, curr_row = fixed_rows[i]

            depth_change = f"{prev_row['hierarchy_depth']}→{curr_row['hierarchy_depth']}"
            delta_dp = curr_row["dendrogram_purity"] - prev_row["dendrogram_purity"]
            delta_ir = curr_row["inheritance_richness"] - prev_row["inheritance_richness"]
            delta_fanout = curr_row["average_fanout"] - prev_row["average_fanout"]

            f.write(
                f"| {depth_change} | "
                f"{delta_dp:+.4f} | "
                f"{delta_ir:+.4f} | "
                f"{delta_fanout:+.4f} |\n"
            )

    print(f"\n{'=' * 70}")
    print("HCR-Stop 深度扫描实验完成")
    print(f"结果JSON: {summary_path}")
    print(f"论文表格: {table_path}")
    print(f"固定深度增量表: {delta_table_path}")
    print(f"{'=' * 70}")

    return results
