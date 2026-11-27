"""Frozen synthetic development and holdout cohorts, separate from Evidence36."""

import argparse
import hashlib
import json
import random
from pathlib import Path

STORES = (
    "松弦",
    "槐序",
    "砚溯",
    "棠屿",
    "墨禾",
    "云瓷",
    "沐榆",
    "照川",
    "榕岸",
    "晴汀",
    "青霁",
    "霜檀",
    "锦笺",
    "南栗",
)
PRODUCTS = (
    "便携咖啡壶",
    "竹席收纳袋",
    "陶艺描金笔",
    "桌面迷你架",
    "雨衣清洁盒",
    "碳钢煎锅",
    "羊毛暖杯套",
    "藤编置物筐",
    "不锈钢搅拌勺",
    "手摇磨豆杯",
    "净水量杯",
    "葡萄木盘",
    "玻璃午餐盒",
    "手冲保温瓶",
    "松石茶筅",
    "竹制蒸笼",
)
TASKS = (
    ("补发运费", "元", ("补寄配件要付多少快递费", "买它以后补寄费用是多少")),
    ("保修期限", "个月", ("保修一共多长时间", "保修承诺是几个月")),
    ("排队时限", "天", ("返修预约最长等几天", "返修预约要等几天")),
    ("自提留存期", "小时", ("门店自取会给我保留多久", "门店最多留货几个小时")),
    ("补配上限", "件", ("最多能给我补配几件", "补配的数量上限是什么")),
)
OUTSIDE = (
    "能帮我写一首写秋天的七言绝句吗？",
    "讲讲太阳系里最大的行星为什么有这么大的风暴。",
    "帮我推导二次函数的求根公式。",
    "解释一下如何用吉他按出 C 大调和弦。",
    "我想知道早晨云层为什么会变成紫色。",
    "给我讲讲城市地下地铁最早出现在哪个年代。",
    "为星期五的读书会推荐一首古诗。",
    "说说海豚用什么方法在水里定位。",
    "说明怎样区分巴洛克和古典主义音乐。",
    "不用代码讲解一下勾股定理的证明。",
    "可以推荐一道番茄炒蛋的家常做法吗？",
    "简述月食为什么不会每个月都发生。",
)
STYLES = ("code", "title", "alias", "sentence")


def stable_id(*parts):
    return "P" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]


def referral(style, title, code, alias, index):
    if style == "code":
        return f"对应政策编号 {code}，请查该编号的现行标准。"
    if style == "title":
        return f"具体约定请参照《{title}》。"
    if style == "alias":
        return f"本项使用内部别名「{alias}」，按该记录核对。"
    return (
        f"查询时使用叫做{alias}的那张独立记录。"
        if index % 2
        else f"名为{alias}的那份说明中有现行规则。"
    )


def create(cohort):
    rng = random.Random(32710 if cohort == "dev" else 92861)
    names = STORES[:2] if cohort == "dev" else STORES[2:]
    groups = (
        ["direct"] * 6 + ["two"] * 8 + ["three"] * 6 + ["missing"] * 2 + ["outside"] * 2
        if cohort == "dev"
        else ["direct"] * 30 + ["two"] * 40 + ["three"] * 30 + ["missing"] * 10 + ["outside"] * 10
    )
    rng.shuffle(groups)
    per_store = len(groups) // len(names)
    cases, docs = [], []
    for store_index, store in enumerate(names):
        products = rng.sample(PRODUCTS, per_store)
        for local, product in enumerate(products):
            index = store_index * per_store + local
            group = groups[index]
            case_id = f"{cohort.upper()}-{index + 1:03}"
            if group == "outside":
                cases.append(
                    dict(
                        id=case_id,
                        store=store,
                        group=group,
                        query=OUTSIDE[index % len(OUTSIDE)],
                        value=None,
                        unit=None,
                        sources=[],
                        decision="outside",
                    )
                )
                continue
            task, unit, wording = TASKS[index % len(TASKS)]
            value = rng.randint(3, 59)
            decision = "missing" if group == "missing" else "answer"
            query = f"我在{store}买的{product}，{wording[rng.randrange(len(wording))]}？"
            depth = {"direct": 1, "two": 2, "three": 3, "missing": 1 + index % 3}[group]
            branch = rng.randrange(90, 991)
            labels = [
                (f"{store}{product}服务索引", f"RV-{branch}01", f"{store}{local:02d}彩签"),
                (f"{store}银页{branch}政策", f"RV-{branch}02", f"{store}{local:02d}竹册"),
                (f"{store}红页{branch}政策", f"RV-{branch}03", f"{store}{local:02d}蓝条"),
            ][:depth]
            sources = []
            for hop, (title, code, alias) in enumerate(labels):
                doc_id = stable_id(cohort, store, case_id, title)
                sources.append(doc_id)
                if hop == depth - 1:
                    body = (
                        f"{task}仍未发布，不能从其它品项的金额、期限或件数推断。"
                        if decision == "missing"
                        else f"现行政策：{task}为{value}{unit}，仅适用该记录对应品项。"
                    )
                    if depth == 1:
                        body = f"{product}的{body}"
                else:
                    next_title, next_code, next_alias = labels[hop + 1]
                    style = STYLES[(index + hop) % len(STYLES)]
                    body = f"本记录覆盖{task}；{referral(style, next_title, next_code, next_alias, index + hop)}"
                    if hop == 0:
                        body = f"{product}的{body}"
                docs.append(
                    dict(
                        id=doc_id,
                        store=store,
                        title=title,
                        text=f"{title}。编号 {code}；别名「{alias}」。{body}",
                    )
                )
            distractor = stable_id(cohort, store, case_id, "distractor")
            title = f"{store}{product}存档问答"
            docs.append(
                dict(
                    id=distractor,
                    store=store,
                    title=title,
                    text=f"{title}。旧版中的{task}为{value + 6}{unit}，于新规则生效后作废。",
                )
            )
            if depth > 1:
                title = f"{store}展示{branch}政策"
                docs.append(
                    dict(
                        id=stable_id(cohort, store, case_id, "similar"),
                        store=store,
                        title=title,
                        text=f"{title}。编号 RV-{branch}09；仅用于样品。{task}为{value + 8}{unit}。",
                    )
                )
            cases.append(
                dict(
                    id=case_id,
                    store=store,
                    group=group,
                    query=query,
                    value=value if decision == "answer" else None,
                    unit=unit if decision == "answer" else None,
                    sources=sources,
                    decision=decision,
                )
            )
    return dict(cohort=cohort, cases=cases, documents=docs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("dev", "holdout"))
    args = parser.parse_args()
    path = Path(__file__).parent / f"{args.cohort}.json"
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(create(args.cohort), output, ensure_ascii=False, indent=2)
        output.write("\n")
