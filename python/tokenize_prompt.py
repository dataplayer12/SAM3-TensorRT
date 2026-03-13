import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tokenize.py 'your prompt text'")
        sys.exit(1)
        
    prompt = sys.argv[1]
    
    # Check if transformers is available
    try:
        from transformers import AutoProcessor
    except ImportError as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print("-1") # Signal failure to C++
        sys.exit(1)
        
    # We load standard SAM3 tokenizer
    # We suppress huggingface warnings so they dont pollute stdout
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        processor = AutoProcessor.from_pretrained("facebook/sam3")
        # Call the text tokenizer directly to bypass the Image/Video extractor requirements
        inputs = processor.tokenizer(text=prompt, return_tensors="pt")
        
        # input_ids is typically shape [1, N]
        ids = inputs.input_ids[0].tolist()
        
        # print comma separated string to stdout for C++ to read
        print(",".join(map(str, ids)))
        
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print("-1") # Signal failure
        sys.exit(1)

if __name__ == "__main__":
    main()
