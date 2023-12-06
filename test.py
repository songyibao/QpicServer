from deoldify import device
from deoldify.device_id import DeviceId
#choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)
from deoldify.visualize import *
plt.style.use('dark_background')
torch.backends.cudnn.benchmark=True
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")


def colorize_image(image_path):
    image_colorizer = get_image_colorizer()  # 使用稳定版颜色处理器，你也可以选择其他的处理器
    # 检查文件是否为图像文件
    if os.path.isfile(image_path) and any(
            image_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']):

        result = image_colorizer.get_transformed_image(path=image_path, render_factor=35, post_process=True,
                                                       watermarked=True)

        if result is not None:
            result.save(image_path, quality=95)
            result.close()


if __name__ == '__main__':
    colorize_image('uploads/1701771694.png')

# colorizer = get_image_colorizer(artistic=True)
# render_factor=35
# #NOTE:  Make source_url None to just read from file at ./video/source/[file_name] directly without modification
# source_url='https://upload.wikimedia.org/wikipedia/commons/e/e4/Raceland_Louisiana_Beer_Drinkers_Russell_Lee.jpg'
# source_path = 'test_images/image.png'
# result_path = None
#
# if source_url is not None:
#     result_path = colorizer.plot_transformed_image_from_url(url=source_url, path=source_path, render_factor=render_factor, compare=True)
# else:
#     result_path = colorizer.plot_transformed_image(path=source_path, render_factor=render_factor, compare=True)
#
# show_image_in_notebook(result_path)