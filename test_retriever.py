from retrieve import HybridRetriever

if __name__ == "__main__":
    R = HybridRetriever()
    query = "Who founded The Statesman?"
    results = R.search(query)   # no top_k arg in your version

    print("Got", len(results), "results\n")

    for i, r in enumerate(results):
        # Try common keys based on how chunks are usually stored
        print(f"--- Result {i+1} ---")
        print("chunk_id:", r.get("chunk_id"))
        print("pdf:", r.get("pdf"))
        print("page:", r.get("page"))
        text = r.get("text", "")
        print("text snippet:", text[:300].replace("\n", " "))
        print()
