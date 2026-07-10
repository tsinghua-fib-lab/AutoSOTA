import matplotlib.pyplot as plt

from .visualization_callback import VisualizationCallback


class ImageLabelVisualizationCallback(VisualizationCallback):
    viz_order = ("y", "img")
    labels = [str(i) for i in range(10)]

    def visualize_y(self, y, y_pred):
        print("Class predictions (in percentage)")

        if y_pred is None:  # "y" didn't need to be predicted
            y_pred = y

        # Print class names
        class_names_str = " ".join(f"{name:^6}" for name in self.labels)
        print(class_names_str)

        # Print predictions
        predictions_percentage = y_pred * 100
        predictions_str = " ".join(
            (
                f"\033[38;2;128;0;0m{pred:^6.1f}\033[0m"
                if idx != y.argmax()
                else f"\033[1m\033[38;2;0;128;0m{pred:^6.1f}\033[0m"
            )
            for idx, pred in enumerate(predictions_percentage)
        )
        print(predictions_str)

    def visualize_img(self, img, img_pred):
        # If no prediction, just show input img
        if img_pred is None:
            img_pred = img

        # Show generated image
        if img_pred.shape[0] == 1:  # only 1 Color channel
            plt.imshow(img_pred.squeeze(0).T.detach(), cmap="gray")
        else:
            plt.imshow(img_pred.permute(1, 2, 0))  # CHW -> HWC
