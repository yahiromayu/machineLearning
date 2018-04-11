import os
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

def draw_images(generator, x, dir_name, index):
    # 出力ファイルの設定
    save_name = 'extened-' + str(index)
    g = generator.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=save_name, save_format='jpg')

    # 1つの入力画像から何枚拡張するかを指定
    # g.next()の回数分拡張される
    for i in range(10):
        bach = g.next()


if __name__ == '__main__':

    # 出力先ディレクトリの設定

    # 拡張する際の設定
    generator = ImageDataGenerator(
                    rotation_range=45, # 90°まで回転
                    width_shift_range=0.1, # 水平方向にランダムでシフト
                    height_shift_range=0.1, # 垂直方向にランダムでシフト
                    channel_shift_range=0, # 色調をランダム変更
                    shear_range=0.39, # 斜め方向(pi/8まで)に引っ張る
                    horizontal_flip=False, # 垂直方向にランダムで反転
                    vertical_flip=True # 水平方向にランダムで反転
                    )

    img_folders = glob.glob(os.path.abspath("./class/*"))
    for i,folder in enumerate(img_folders):
        img_list = glob.glob(folder+"/*.jpg")
        output_dir = folder+"/extended"
        if not(os.path.exists(output_dir)):
            os.mkdir(output_dir)
        for j,img_path in enumerate(img_list):
            img = img_to_array(load_img(img_path))
            img = np.expand_dims(img,axis=0)
            draw_images(generator, img, output_dir, j)
