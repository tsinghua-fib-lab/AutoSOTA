from .drmetrics.imbalance_degree import imbalance_degree


def generate_readiness_report(train_datasets, dr_metrics):
    print("------- Data Readiness Report - Begin --------")
    if "ci" in dr_metrics:
        print("Class Imbalance Degree:")
        for i, dataset in enumerate(train_datasets):
            # Access the labels from the current dataset
            labels = dataset.data_label.tolist()
            imb_deg = imbalance_degree(labels)
            print(f"  Client {i + 1}: {round(imb_deg, 2)}")

    if "ss" in dr_metrics:
        print("Sample Size:")

        for i, dataset in enumerate(train_datasets):
            sample_size = len(dataset.data_label.tolist())
            print(f"  Client {i + 1}: {round(sample_size)}")

    if "rr" in dr_metrics:
        print("Representation Rate:")
        for i, dataset in enumerate(train_datasets):
            gender_list = dataset.data_gender.tolist()
            count_1s = gender_list.count(1)
            count_0s = gender_list.count(0)
            total_samples = len(gender_list)
            representation_rate_1s = count_1s / total_samples
            representation_rate_0s = count_0s / total_samples
            print(
                f"  Client {i + 1}: Representation Rate of Males: {round(representation_rate_1s, 4)}, Representation Rate of Females: {round(representation_rate_0s, 4)}"
            )

    print("------- Data Readiness Report - End ----------")
