# RegText

## Abstract
The widespread practice of indiscriminate data scraping to fine-tune language models (LMs) raises significant legal and ethical concerns, particularly regarding compliance with data protection laws such as the General Data Protection Regulation (GDPR). This practice often results in the unauthorized use of personal information, prompting growing debate within the academic and regulatory communities. Recent works have introduced the concept of generating unlearnable datasets (by adding imperceptible noise to the clean data), such that the underlying model achieves lower loss during training but fails to generalize to the unseen test setting. Though somewhat effective, these approaches are predominantly designed for images and are limited by several practical constraints like requiring knowledge of the target model. To this end, we introduce RegText, a framework that injects imperceptible spurious correlations into natural language datasets, effectively rendering them unlearnable without affecting semantic content. We demonstrate RegText's utility through rigorous empirical analysis of small and large LMs. Notably, RegText can restrict newer models like GPT-4o and Llama from learning on our generated data, resulting in a drop in their test accuracy compared to their zero-shot performance and paving the way for generating unlearnable text to protect public data.

Read the full paper accepted at NAACL 2025 [here](https://openreview.net/forum?id=5QRQd3uVFs).

## Key Features
- **Imperceptible Noise Injection:** Adds spurious correlations to text data without altering semantic meaning.
- **Unlearnable Text Datasets:** Prevents LMs from generalizing effectively on the modified datasets.
- **Broad Compatibility:** Demonstrated effectiveness on both small and large language models, including state-of-the-art systems like GPT-4o and Llama.
- **Empirical Validation:** Rigorous testing confirms significant drops in test accuracy compared to zero-shot performance.

## Installation and Setup
To reproduce the results outlined in our paper, follow these steps:

### 1. Clone the Natural Instructions Repository
```bash
pip install -r requirements.txt
cd create_data
git clone https://github.com/allenai/natural-instructions.git
cp -r polarity/ natural-instructions/splits/
```

### 2. Generate Data 
Run the following commands to generate the poisoned datasets for, 
1. Natural Instructions Polarity Dataset:
```bash
python generate_polarity <metric-type ("global")> <seed (0, 1, 42)> <max_poisons (10)>
python3 convert_to_csv.py
```

2. Imdb or AGNews: 
```bash
python generate <seed (0, 1, 42)> <max_poisons (10)> <dataset_path (../data/imdb)>
```

> **Note:** Adjust the configurations in the scripts if required.

### 3. Train Models and Evaluate
Navigate to the evaluation directory and run the training and testing scripts:
```bash
cd ../evaluation

# Train the model
python train.py --file-path <path-to-training-file> \
                --test-file-path <path-to-test-file> \
                --model-name <model-name> \
                --hf-token <Hugging Face token> \
                --save-dir <directory-to-save-model>

# Evaluate the model
python test.py --file-path <path-to-training-file> \
               --test-file-path <path-to-test-file> \
               --model-name <model-name> \
               --hf-token <Hugging Face token> \
               --save-dir <directory-to-save-model> \
               --ckpt-dir <path-to-checkpoint-directory>
```




## Contributing
We welcome contributions to enhance RegText! If you have suggestions or find issues, please submit a pull request or open an issue in this repository.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
We extend our gratitude to the developers of the [Natural Instructions Repository](https://github.com/allenai/natural-instructions) for providing the foundation for our dataset generation pipeline.
