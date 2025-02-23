from src.interference import ModelInterference

def test_model_interference(data_path = None):
    try:
        model_interference = ModelInterference(data_path)
        model_interference.run()  

        print("Model interference completed successfully.")
    
    except Exception as e:
        print(f"Error in model interference: {e}")

if __name__ == "__main__":
    print("Starting Model Interference test...\n")
    
    test_model_interference()
