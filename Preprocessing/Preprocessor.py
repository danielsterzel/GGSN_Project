

IMG_SIZE = 256


class Preprocessor:

    def __init__(self, resize=IMG_SIZE, normalize_function=None):
        self.resize = resize
        self.normalize_function = normalize_function

    @staticmethod
    def patchify(img, patch_size, stride):
        import numpy as np
        patches = []

        h, w, _ = img.shape

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):

                patch = img[y:y + patch_size, x: x + patch_size]
                patches.append(patch)

        return np.array(patches)


    def __call__(self, img):
        if self.resize is not None:
            import cv2
            img = cv2.resize(img, (self.resize, self.resize))


        if self.normalize_function:
            img = self.normalize_function(img)

        return img