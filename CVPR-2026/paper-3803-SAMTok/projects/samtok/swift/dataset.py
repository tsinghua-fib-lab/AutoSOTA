from swift.llm import MessagesPreprocessor, DatasetMeta, register_dataset


register_dataset(
    DatasetMeta(
        dataset_name="mask_generation_gres",
        dataset_path="zhouyik/SAMTok_Training_Data/mask_generation_gres209k.json",
        preprocess_func=MessagesPreprocessor(),
    )
)
