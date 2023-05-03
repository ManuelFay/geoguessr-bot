import gradio as gr
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()

metadata = pd.read_csv("metadatav3.csv")
metadata.path = metadata.path.apply(lambda x: x.split("/")[-1])


embeddings = np.load("embeddings.npy")
test_embeddings = np.load("test_embeddings.npy")
files = open("files.txt").read().split("\n")
test_files = open("test_files.txt").read().split("\n")
print(embeddings.shape, test_embeddings.shape, len(files), len(test_files))

knn = NearestNeighbors(n_neighbors=50, algorithm='kd_tree', n_jobs=8)
knn.fit(embeddings)

# %%
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])


def cluster(df, eps=0.1, min_samples=5, metric="cosine", n_jobs=8, show=False):
    if len(df) == 1:
        return df
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=n_jobs)
    dbscan.fit(df[["longitude", "latitude"]])
    df["cluster"] = dbscan.labels_
    # Return centroid of the cluster with the most points
    df = df[df.cluster == df.cluster.value_counts().index[0]]
    df = df.groupby("cluster").apply(lambda x: x[["longitude", "latitude"]].median()).reset_index()
    # Return coordinates of the cluster with the most points
    return df.longitude.iloc[0], df.latitude.iloc[0]


def guess_image(img):
    # img = Image.open(image_path)
    # cast as rgb
    img = img.convert('RGB')
    print(img)
    with torch.no_grad():
        features = model(transform(img).unsqueeze(0))[0].cpu()
        distances, neighbors = knn.kneighbors(features.unsqueeze(0))

    neighbors = neighbors[0]
    # Return metadata df rows with neighbors
    df = pd.DataFrame()
    for n in neighbors:
        df = pd.concat([df, metadata[metadata.path == files[n]]])
    coords = cluster(df, eps=0.005, min_samples=5)

    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = GeoDataFrame(df, geometry=geometry)
    gdf_guess = GeoDataFrame(df[:1], geometry=[Point(coords)])
    # this is a simple map that goes with geopandas
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    plot_ = world.plot(figsize=(10, 6))
    gdf.plot(ax=plot_, marker='o', color='red', markersize=15)
    gdf_guess.plot(ax=plot_, marker='o', color='blue', markersize=15);
    return coords, plot_.figure


# Image to image translation
def translate_image(input_image):
    coords, fig = guess_image(Image.fromarray(input_image.astype('uint8'), 'RGB'))
    fig.savefig("tmp.png")
    return str(coords), np.array(Image.open("tmp.png").convert("RGB"))


demo = gr.Interface(fn=translate_image, inputs="image", outputs=["text", "image"], title="Street View Location", description="Helps you guess the location of a street view image ! Use it on square images with no goole maps artefacts when possible !")

if __name__ == "__main__":
    demo.launch()
