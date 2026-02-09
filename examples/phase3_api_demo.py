"""Phase 3.2: API and Dashboard Demo.

Demonstrates:
1. Starting FastAPI server
2. Making predictions via REST API
3. Data loading and preprocessing
4. End-to-end pipeline
"""

import time

import requests


def test_api_health():
    """Test API health endpoint."""
    print("=" * 60)
    print("Testing API Health")
    print("=" * 60)

    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy!")
            print(response.json())
        else:
            print(f"❌ API returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is it running?")
        print("\nStart API with: uvicorn phio.api.app:create_app --factory")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_prediction_api():
    """Test prediction endpoint."""
    print("\n" + "=" * 60)
    print("Testing Prediction API")
    print("=" * 60)

    # Prepare request
    request_data = {
        "x": [0.1, 0.2, 0.3, 0.4, 0.5],
        "t": [0.0, 0.0, 0.0, 0.0, 0.0],
        "model_name": "heat-1d-demo",
    }

    print("\nRequest:")
    print(f"  X points: {request_data['x']}")
    print(f"  Time: {request_data['t'][0]}")

    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=request_data,
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            print("\n✅ Prediction successful!")
            print(f"  Points predicted: {data['n_points']}")
            print(f"  Model version: {data['model_version']}")
            print(f"  Predictions: {data['predictions'][:3]}...")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.json())

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API")
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_data_loading():
    """Demo data loading pipeline."""
    print("\n" + "=" * 60)
    print("Data Loading Demo")
    print("=" * 60)

    from phio.data import DataLoader, create_collocation_points
    import jax
    import jax.numpy as jnp

    # Create sample data
    print("\nGenerating sample data...")
    rng = jax.random.PRNGKey(42)

    domain = {
        "x": (0.0, 1.0),
        "y": (0.0, 1.0),
        "t": (0.0, 1.0),
    }

    points = create_collocation_points(
        domain=domain,
        n_pde=100,
        n_bc=20,
        n_ic=20,
        rng=rng,
    )

    print("\n✅ Collocation points generated:")
    for key, value in points.items():
        print(f"  {key}: shape {value.shape}")

    # Save data
    loader = DataLoader()
    loader.save(points, "sample_data.npz", format=".npz")
    print("\n✅ Data saved to sample_data.npz")

    # Load data
    loaded = loader.load("sample_data.npz")
    print("\n✅ Data loaded successfully:")
    for key in loaded.keys():
        print(f"  {key}: {loaded[key].shape}")


def demo_normalization():
    """Demo data normalization."""
    print("\n" + "=" * 60)
    print("Data Normalization Demo")
    print("=" * 60)

    from phio.data import Normalizer
    import jax.numpy as jnp

    # Sample data
    data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"\nOriginal data: {data}")

    # Min-max normalization
    normalizer = Normalizer(method="minmax")
    normalized = normalizer.fit_transform(data)
    print(f"Normalized (minmax): {normalized}")

    # Inverse transform
    recovered = normalizer.inverse_transform(normalized)
    print(f"Recovered: {recovered}")
    print(f"Error: {jnp.max(jnp.abs(data - recovered)):.2e}")


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("# PHASE 3.2: END-TO-END PIPELINE DEMO")
    print("#" * 60)

    print("\nThis demo showcases:")
    print("  1. FastAPI REST API")
    print("  2. Data loading and preprocessing")
    print("  3. Normalization")
    print("  4. Docker deployment (see docker-compose.yml)")

    # Data demos (always work)
    demo_data_loading()
    demo_normalization()

    # API demos (require server running)
    print("\n" + "=" * 60)
    print("API Demos (requires server running)")
    print("=" * 60)
    print("\nTo start the API server:")
    print("  uvicorn phio.api.app:create_app --factory --reload")
    print("\nOr with Docker:")
    print("  docker-compose up")
    print("\nThen visit:")
    print("  - API docs: http://localhost:8000/docs")
    print("  - Dashboard: http://localhost:8501")

    print("\nAttempting API connection...")
    time.sleep(1)

    test_api_health()
    test_prediction_api()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Start API: uvicorn phio.api.app:create_app --factory")
    print("  2. Start Dashboard: streamlit run dashboard/app.py")
    print("  3. Or use Docker: docker-compose up")


if __name__ == "__main__":
    main()
