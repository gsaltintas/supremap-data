import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from osmCategories import *
import shapely
from pathlib import Path
import cairosvg
import numpy as np
import io
# install osmnx through pip, conda version is too old
# osmnx version should be 1.2.2 and shapely 2.0.0
import osmnx as ox

import matplotlib as mpl
# for running without GUI
mpl.use('Agg')


def plot_seg_and_inst_map(bbox, save_dir, filename, resolution=3000, background_color=(255, 255, 255, 255),
                          street_widths={
    "footway": 1.5,
    "steps": 1.5,
    "pedestrian": 1.5,
    "service": 1.5,
    "path": 1.5,
    "track": 1.5,
    "motorway": 6,
    "default": 5,
}):
    """creates and saves osm segmentation and instancing map within bbox to save_dir
        bbox format: [lat1, lat2, long1, long2] or [north, south, east, west], N and S can be inverted

        save_dir is a Path object

        street_widths is scaled by resolution, thus independant
    """

    # scale roadWidth by resolution
    scale = street_widths['default'] / 2000 / 4 * resolution
    for key in street_widths:
        street_widths[key] *= scale

    if bbox is None:
        bbox = [47.368614, 47.377607, 8.53769, 8.55093]
        save_dir = Path("./osm").resolve().absolute()
        filename = "output"

    # sort bounding box
    bbox = list(bbox)
    bbox[0:2] = sorted(bbox[0:2])
    bbox[2:4] = sorted(bbox[2:4])

    # convert bbox to poly
    poly = shapely.geometry.box(
        *(bbox[3], bbox[1], bbox[2], bbox[0]), ccw=True)

    plt.ioff()

    # function to create lots of colors
    def getColormap(currOffset, currAmount):
        return ListedColormap([f'#{color:06x}' for color in range(currOffset, currOffset + currAmount)])

    def plotRoads(plotSeg, plotInst, offset, tags, color, street_widths):
        roads = ox.graph_from_polygon(
            poly, retain_all=True, truncate_by_edge=True)
        geo = ox.utils_graph.graph_to_gdfs(roads, nodes=False)

        for index, (type, size) in enumerate(street_widths.items()):
            typeGeo = geo.loc[geo['highway'] == type]
            amount = 2 * typeGeo.shape[0]
            typeGeo.plot(ax=plotSeg[index][1], color=color, linewidth=size)
            typeGeo.plot(ax=plotInst[index][1], cmap=getColormap(
                offset, amount), linewidth=size)
            offset = offset + amount

        typeGeo = geo.loc[~geo['highway'].isin(street_widths.keys())]
        size = street_widths['default']
        amount = 2 * typeGeo.shape[0]
        typeGeo.plot(ax=plotSeg[-1][1], color=color, linewidth=size)
        typeGeo.plot(ax=plotInst[-1][1], cmap=getColormap(
            offset, amount), linewidth=size)
        offset = offset + amount

        return offset

    def plotGeometry(axisSeg, axisInst, offset, tags, color):
        geo = ox.geometries_from_polygon(
            poly, tags=tags)
        amount = 2 * geo.shape[0]
        geo.plot(ax=axisSeg, color=color)
        geo.plot(ax=axisInst, cmap=getColormap(offset, amount))
        return offset + amount

    # arrays to save layers of osm images
    imagesSeg = []
    imagesInst = []

    # offset for colormap creation
    offset = 0

    for index, (tag, color) in enumerate([(WaterTags, 'blue'), (FieldTags, 'green'), (BeachTags, 'brown'), ("roads", 'red'), (BuildingTags, 'black')]):
        figSeg, axSeg = plt.subplots(figsize=(10, 10))
        figInst, axInst = plt.subplots(figsize=(10, 10))

        plotSeg = [(figSeg, axSeg)]
        plotInst = [(figInst, axInst)]

        # draw content
        if (tag == "roads"):
            # create an additional plot for each street width option
            for _ in street_widths:
                plotSeg.append(plt.subplots(figsize=(10, 10)))
                plotInst.append(plt.subplots(figsize=(10, 10)))

            print(f'Fetching and plotting layer {index} with roads')
            offset = plotRoads(plotSeg, plotInst, offset,
                               tag, color, street_widths)

        else:
            print(f'Fetching and plotting layer {index} with {tag.keys()}')
            offset = plotGeometry(axSeg, axInst, offset, tag, color)
            figSeg = []

        # convert ax to svg to png
        for plots, outputArr in [(plotSeg, imagesSeg), (plotInst, imagesInst)]:
            for fig, ax in plots:
                # crop
                ax.set_xlim((bbox[2], bbox[3]))
                ax.set_ylim((bbox[0], bbox[1]))
                ax.set_axis_off()

                fig.tight_layout(pad=-0.08)

                # convert plot to svg, then save svg without anti-aliasing
                imgdata = io.StringIO()
                fig.savefig(imgdata, format='svg', transparent=True)
                plt.close(fig)
                imgdata.seek(0)  # rewind the data

                svg_dta = imgdata.read()  # convert to string

                # add no antialiasing to svg
                svg_dta = svg_dta.replace(
                    "version=\"1.1\">", "shape-rendering=\"crispEdges\" version=\"1.1\">")

                # save png to buffer, load with PIL and append to images
                buffer = io.BytesIO()
                cairosvg.svg2png(bytestring=svg_dta,
                                 write_to=buffer, output_height=resolution)

                buffer.seek(0)
                outputArr.append(Image.open(buffer))

    # stack images and save to PNG file
    background = Image.new(
        mode="RGBA", size=(resolution, resolution), color=background_color)
    for image in imagesSeg:
        background = Image.alpha_composite(background, image)

    save_dir.mkdir(exist_ok=True, parents=True)

    print(f'saving to {save_dir.joinpath(f"{filename}_TYPE.png")}')
    background.save(save_dir.joinpath(
        f'{filename}_segmentation.png'), "PNG")

    background = Image.new(
        mode="RGBA", size=(resolution, resolution), color=background_color)
    for image in imagesInst:
        background = Image.alpha_composite(background, image)

    background.save(save_dir.joinpath(
        f'{filename}_instantation.png'), "PNG")


# for running as a file
def main():
    plot_seg_and_inst_map(None, None, None)


if __name__ == "__main__":
    main()