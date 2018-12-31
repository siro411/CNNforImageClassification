from pathlib import Path
from tqdm import tqdm
import numpy as np
import scipy.io
import cv2



def main():
    image_dir = Path("/Users/annaying/final project/UTKFace")
    output_path = "/Users/annaying/final project/UTKFace"
    img_size = 64

    out_genders = []
    out_ages = []
    out_imgs = []

    for i, image_path in enumerate(tqdm(image_dir.glob("*.jpg"))):
        image_name = image_path.name  # [age]_[gender]_[race]_[date&time].jpg
        age, gender = image_name.split("_")[:2]
        out_genders.append(int(gender))
        out_ages.append(min(int(age), 100))
        img = cv2.imread(str(image_path))
        # out_imgs.append(cv2.resize(img, (img_size, img_size)))  #BGR in opencv
        out_imgs.append(cv2.cvtColor(cv2.resize(img, (img_size, img_size)), cv2.COLOR_BGR2RGB)) #RGB
    output = {"image": np.array(out_imgs), "gender": np.array(out_genders), "age": np.array(out_ages),
              "db": "utk", "img_size": img_size, "min_score": -1}
    scipy.io.savemat(output_path, output)


if __name__ == '__main__':
    main()
