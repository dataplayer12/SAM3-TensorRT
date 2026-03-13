from transformers import AutoProcessor
import os

def main():
    if not os.path.exists('onnx_weights'):
        os.makedirs('onnx_weights')
        
    print("Loading SAM3 Processor...")
    processor = AutoProcessor.from_pretrained("facebook/sam3")
    
    print("Saving tokenizer configuration to onnx_weights/tokenizer.json")
    processor.tokenizer.save_pretrained("onnx_weights/")
    print("Done!")

if __name__ == "__main__":
    main()
