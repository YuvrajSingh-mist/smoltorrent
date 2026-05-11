import httpx

API_BASE = "http://localhost:8000"


def gather_shards() -> dict:
    resp = httpx.post(f"{API_BASE}/gather-shards", timeout=300.0)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    print("Triggering shard gather...")
    result = gather_shards()
    print(f"Gathered {len(result['gathered'])} shards → {result['save_path']}")


if __name__ == "__main__":
    main()
