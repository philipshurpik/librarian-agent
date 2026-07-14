"""Retrieval eval: golden-query recall@3 + calibration of the weak-match score threshold.

Needs live OpenAI + Qdrant with an ingested catalog: `make ingest && make eval`.
"""

import asyncio
import sys

from sklearn.metrics import fbeta_score, precision_recall_curve

from librarian.agent.tools import search_catalog

GOLDEN = {
    'exactly-once semantics in stream processing': 'Streaming Systems',
    'how do B-tree indexes work': 'SQL Performance Explained',
    'designing distributed data systems': 'Designing Data-Intensive Applications',
    'practical machine learning': 'Hands-On Machine Learning',
    'writing clean readable code': 'Clean Code',
    'container orchestration with kubernetes': 'Kubernetes: Up and Running',
    'tuning mysql database performance': 'High Performance MySQL',  # no-description book: metadata-fallback path
}
OFF_TOPIC = [
    'best recipes for italian pasta',
    'history of the roman empire',
    'iOS development with Swift',
    'excel spreadsheet formulas',
    'monolith buildings construction',
    'Kotlin Programming language',
]


def calibrate_threshold(relevant: list[float], off_topic: list[float]) -> bool:
    """Select the F2-best cut band (recall weighted over precision): a junk pass-through is recoverable
    downstream (LLM reads snippets, future reranker), a real match wrongly flagged as weak is not.
    """
    scores, labels = relevant + off_topic, [1] * len(relevant) + [0] * len(off_topic)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f2s = [fbeta_score(labels, [s >= t for s in scores], beta=2) for t in thresholds]
    bands = list(zip([0.0, *thresholds], thresholds))  # all cuts inside one band classify identically
    for (lo, hi), p, r, f in zip(bands, precision, recall, f2s):
        print(f'threshold in ({lo:.3f}, {hi:.3f}]: precision {p:.0%}, recall {r:.0%}, f2 {f:.2f}')

    lo, hi = max(zip(f2s, bands))[1]
    print(f'selected midpoint {(lo + hi) / 2:.3f} of the F2-best band ({lo:.3f}, {hi:.3f}]')
    print(f'max f2 {max(f2s):.2f} (gate: >= 0.9)')
    return max(f2s) >= 0.9


async def main() -> int:
    print('--- golden queries (expected book in top 3) ---')
    misses, relevant_scores = [], []
    for query, expected in GOLDEN.items():
        hits = await search_catalog(query, limit=3)
        rank = next((i for i, h in enumerate(hits) if expected.lower() in h['title'].lower()), None)
        relevant_scores.append(hits[0]['score'])
        misses.extend([query] if rank is None else [])
        status = 'MISS  ' if rank is None else f'rank {rank + 1}'
        print(f'{status} | top {hits[0]["score"]:.3f} {hits[0]["title"][:42]:42} | {query}')

    print('--- off-topic queries (expect scores below the threshold) ---')
    off_scores = []
    for query in OFF_TOPIC:
        top = (await search_catalog(query, limit=1))[0]
        off_scores.append(top['score'])
        print(f'       | top {top["score"]:.3f} {top["title"][:42]:42} | {query}')

    print(f'\nrecall@3: {len(GOLDEN) - len(misses)}/{len(GOLDEN)}')
    ok = calibrate_threshold(relevant_scores, off_scores) and not misses
    print('OK' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
