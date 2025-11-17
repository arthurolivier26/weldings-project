import pandas as pd
import tensorflow as tf
from pathlib import Path


def build_tf_dataset_from_df(
    df: pd.DataFrame,
    root_dir: str | Path = ".",
    path_col: str = "path",
    label_col: str = "label",
    batch_size: int = 32,
    shuffle: bool = True,
    img_size: tuple[int, int] = (224, 224),
    num_classes: int | None = None,
):
    """
    Construit un tf.data.Dataset à partir d'un DataFrame.

    - root_dir : préfixe des chemins si `path` est relatif
    - img_size : taille de resize (H, W)
    - num_classes : si fourni, on renvoie des labels one-hot, sinon labels entiers
    """

    root_dir = Path(root_dir)

    # chemins absolus sous forme de strings
    paths = [str(root_dir / p) for p in df[path_col].values]
    labels = df[label_col].values

    # Conversion en tensors
    paths_tf = tf.constant(paths)
    labels_tf = tf.constant(labels)

    ds = tf.data.Dataset.from_tensor_slices((paths_tf, labels_tf))

    def _load_image(path, label):
        # Lire l'image
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]

        # Resize
        img = tf.image.resize(img, img_size)

        # Ici tu peux ajouter d'autres augmentations / préprocessing TF si tu veux
        # ex: img = tf.image.random_flip_left_right(img)

        # Label : optionnel one-hot
        if num_classes is not None:
            label_int = tf.cast(label, tf.int32)
            label = tf.one_hot(label_int, depth=num_classes)

        return img, label

    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def test_dataloader(path_data='../notebooks_cache'):
    df_train = dm.explore_csv_hierarchy(path_data,allowed_ext='.jpeg')
    mapping = {'OK': 0, 'KO': 1}
    df_train['label'] = df_train['level_4'].map(mapping)
    train_ds = build_tf_dataset_from_df(df=df_train,
                                        root_dir="../Challenge-Welding-Reference-Solution1/",
                                        path_col="path",label_col="label",batch_size=32,img_size=(224, 224),num_classes=2)
    
    for batch_idx, (images, labels) in enumerate(train_ds):
        print("Batch index :", batch_idx)
        print("Images shape :", images.shape)   # ex: (32, 224, 224, 3)
        print("Labels shape :", labels.shape)   # ex: (32,) ou (32, num_classes)
        print("Labels :", labels.numpy())
    
        # Afficher un seul batch puis stopper
        break