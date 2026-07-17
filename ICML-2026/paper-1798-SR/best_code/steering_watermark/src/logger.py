## In this script is added the elements relative to the potentially distant logger
# Such as the clearml

def extract_logs_from_df(df, task):

    # print(df.columns)
    # exit()

    df_light = df.drop(columns=["fwd_data", "activations"], inplace=False)

    task.upload_artifact(name="metrics_dataframe", artifact_object=df_light)