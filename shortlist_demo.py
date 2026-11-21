from src.shortlist import ResumeShortlister

if __name__ == "__main__":
    s = ResumeShortlister()
    s.ensure("data")   # first run builds the index (slow); later runs load it

    results = s.shortlist(
        query="DevOps engineer who builds CI/CD pipelines and manages cloud infrastructure",
        must_have=["aws", "terraform"],
        nice_to_have=["kubernetes", "docker", "jenkins"],
        top_n=10,
    )

    for rank, r in enumerate(results, 1):
        print(f"{rank:>2}. {r['source_file']:<40} score={r['score']:>5}  "
              f"(keywords {r['keyword_score']}, semantic {r['semantic_score']})")
        print(f"    must: {r['matched_must']}  missing: {r['missing_must']}  nice: {r['matched_nice']}")