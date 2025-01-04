import torch
import os
from functools import lru_cache
import numpy as np


script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class CollectImageSize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_image_size"

    def get_image_size(self, image):
        return (image.shape[2], image.shape[1],)

class NearestSDXLResolutionby64:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT",),
                "height": ("INT",),
                "max_pixel_count": ("INT", {"default": 1024}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "calculate"

    def calculate(self, width, height, max_pixel_count, image=None):
        if image != None:
            width = image.shape[2]
            height = image.shape[1]

        image_pixels = width * height
        dest_image_pixels = max_pixel_count ** 2

        factor = (image_pixels / dest_image_pixels) ** 0.5
        new_width = int(int(width / factor) / 64) * 64
        new_height = int(int(width / factor) / 64) * 64

        print(f"New width: {new_width}, New height: {new_height}")

        return (new_width, new_height,)

class GetImageMainColor:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "colors": ("INT", {"default": 5}),
            }
        }
    RETURN_TYPES = ("LIST", "LIST",)
    RETURN_NAMES = ("Color_RGB", "Color String",)

    FUNCTION = "get_main_color"

    MAX_LEVEL = 8

    def get_color_name(self, color):
        # Convert the color to HSV
        try:
            import colorsys
        except ImportError:
            raise ImportError("Please install colorsys to use the string function in the node 'GetImageMainColor'.")

        color_string = []

        for color_tuple in color:
            h, s, v = colorsys.rgb_to_hsv(color_tuple[0] / 255.0, color_tuple[1] / 255.0, color_tuple[2] / 255.0)

            # Determine the lightness based on the v value
            lightness = "dark" if v < 0.3 else "light" if v > 0.6 else "medium"

            # Determine the hue based on the h value
            if h < 0.16:
                hue = "red"
            elif h < 0.33:
                hue = "yellow"
            elif h < 0.5:
                hue = "green"
            elif h < 0.66:
                hue = "cyan"
            elif h < 0.83:
                hue = "blue"
            else:
                hue = "magenta"

            # Determine the saturation based on the s value
            saturation = "low" if s < 0.3 else "high" if s > 0.6 else "medium"
            color_string.append(f"{lightness} {hue} with {saturation} saturation")

        return color_string

    def color_add(self, color1, color2):
        """将两个颜色相加"""
        return (color1[0] + color2[0], color1[1] + color2[1], color1[2] + color2[2])

    def color_div(self, color, k):
        """将颜色除以k"""
        if k == 0:
            raise ValueError("不能除以零")
        return (color[0] // k, color[1] // k, color[2] // k)

    def get_color_index(self, color, level):
        """根据八叉树原理获取颜色索引"""
        r = format(color[0], '08b')[level]
        g = format(color[1], '08b')[level]
        b = format(color[2], '08b')[level]
        return int(r + g + b, 2)

    def normalize_color(self, color, count):
        """对颜色进行归一化"""
        return self.color_div(color, count)

    # 八叉树节点功能
    def create_node(self, level):
        """创建一个新的节点"""
        return {
            'color': (0, 0, 0),  # 初始颜色
            'level': level,  # 节点所在层级
            'children': [None] * 8,  # 子节点
            'pixel_count': 0  # 节点像素数
        }

    def add_color_to_node(self, node, color, level):
        """将颜色添加到节点"""
        if level < self.MAX_LEVEL:
            index = self.get_color_index(color, level)
            if node['children'][index] is None:
                node['children'][index] = self.create_node(level + 1)  # 创建子节点
            self.add_color_to_node(node['children'][index], color, level + 1)
        else:
            node['color'] = self.color_add(node['color'], color)
            node['pixel_count'] += 1

    def reduce_node(self, node):
        """合并节点的孩子"""
        reduce_count = 0
        for i in range(8):
            if node['children'][i] is not None:
                node['color'] = self.color_add(node['color'], node['children'][i]['color'])
                node['pixel_count'] += node['children'][i]['pixel_count']
                reduce_count += 1
        node['children'] = [None] * 8
        return reduce_count - 1  # 返回合并的子节点数量，减去1是因为本节点自己也变成叶子节点

    def get_leaf_nodes(self, node):
        """获取节点及其子树中的所有叶节点"""
        leaf_nodes = []
        if node['pixel_count'] > 0:  # 叶节点
            leaf_nodes.append(node)
        else:
            for child in node['children']:
                if child is not None:
                    leaf_nodes.extend(self.get_leaf_nodes(child))
        return leaf_nodes

    # 八叉树提取颜色
    def extract_colors_from_octree(self, root, k=256):
        """从八叉树中提取k个颜色"""
        leaf_count = len(self.get_leaf_nodes(root))  # 获取叶节点的数量
        for i in range(self.MAX_LEVEL, 0, -1):  # 从level 7开始
            if leaf_count <= k:
                break
            for node in self.get_leaf_nodes(root):
                leaf_count -= self.reduce_node(node)
                if leaf_count <= k:
                    break

        colors = []
        leaf_nodes = self.get_leaf_nodes(root)
        for node in leaf_nodes:
            if node['pixel_count'] > 0:
                colors.append(self.normalize_color(node['color'], node['pixel_count']))
                if len(colors) >= k:
                    break
        return colors

    def extract_dominant_colors(self, image: torch.Tensor, num_colors: int):
        """
        提取图片的主题色。

        参数：
            image (torch.Tensor): 输入图片，形状为 (1, h, w, c)，类型为 torch.Tensor。
            num_colors (int): 要提取的主题色数量。

        返回：
            List[Tuple[int, int, int]]: 提取的主题色列表，每个颜色以 (r, g, b) 表示。
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("image 必须是 torch.Tensor 类型。")
        if image.dim() != 4 or image.size(0) != 1:
            raise ValueError("image 必须具有形状 (1, h, w, c)。")
        if not isinstance(num_colors, int) or num_colors <= 0:
            raise ValueError("colors 必须是一个正整数。")

        # 将 torch.Tensor 转换为 NumPy 数组，并转换为 uint8 类型
        img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1 else img_np.astype(np.uint8)

        h, w, c = img_np.shape
        if c < 3:
            raise ValueError("图片必须至少有 3 个颜色通道（RGB）。")

        # 创建八叉树的根节点
        root = self.create_node(-1)

        # 将图片中的每个像素添加到八叉树中
        for i in range(h):
            for j in range(w):
                r, g, b = img_np[i, j, :3]
                self.add_color_to_node(root, (int(r), int(g), int(b)), 0)

        # 提取颜色
        colors = self.extract_colors_from_octree(root, num_colors)

        return colors

    def get_main_color(self, image, colors):
        try:
            import torch
        except ImportError:
            raise ImportError("Please install torch to use this node.")

        colors_array = self.extract_dominant_colors(image, colors)

        print(colors_array)
        color_list = self.get_color_name(colors_array)
        colors_array_list = [list(color) for color in colors_array]

        return colors_array_list, color_list

class CheckAnimeCharacter:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tags": ("STRING",),
                "chara_tag": ("STRING",),
                "fault_tolerant_num": ("INT",{"default": 3})
            },
            "optional": {
                "reload_character_tags": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING",)
    RETURN_NAMES = ("BOOLEAN", "TAGS",)
    FUNCTION = "check_character_tag"

    tag_map_file = os.path.join(script_directory, "tag_infos.json")

    def reformat_chara_tag(self, tag):
        if '/' or '\\' in tag:
            tag = tag.replace('/', '').replace('\\', '')
        if ' ' in tag:
            tag = tag.replace(' ', '_')
        return tag

    def get_core_tags(self, character_tag):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Please install pandas to use node 'CheckCharacter'.")
        tag_map = pd.read_json(os.path.join(script_directory, 'ComfyUI-PDiD-Nodes', 'index', "tag_infos.json"))
        filtered_data = tag_map[tag_map['tag'] == character_tag]
        if not filtered_data.empty:
            return filtered_data['core_tags']
        else:
            raise ValueError(f"Character tag '{character_tag}' not found in tag map.")

    def check_character_tag(self, tags, chara_tag, fault_tolerant_num, reload_character_tags=False):
        if not os.path.exists(os.path.join(script_directory, 'ComfyUI-PDiD-Nodes', "tag_infos.json")) or reload_character_tags:
            try:
                import requests
                try:
                    response = requests.get("https://huggingface.co")
                    response.raise_for_status()
                except Exception as e:
                    print(f"Failed to connect to Hugging Face: {e}.\n Using mirror site.")
                    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                import huggingface_hub
                huggingface_hub.hf_hub_download(
                    repo_id="deepghs/character_index",
                    filename="index/tag_infos.json",
                    local_dir=os.path.join(script_directory, 'ComfyUI-PDiD-Nodes'),
                    repo_type='dataset')
            except ImportError:
                raise ImportError("Please install huggingface_hub and requests to use node 'CheckCharacter'.")
            except Exception as e:
                raise Exception(f"Failed to download tag_map.json: {e}")

        character_tag = self.reformat_chara_tag(chara_tag)

        core_tags = self.get_core_tags(character_tag) # 'pandas.core.series.Series'
        # Convert the Series to a list
        core_tags = core_tags.tolist()[0]

        core_tags_string = ', '.join(core_tags)
        tags_list = tags.split(', ')
        for core_tag in core_tags:
            if core_tag in tags_list:
                continue
            else:
                fault_tolerant_num -= 1
                if fault_tolerant_num == 0:
                    return (False, core_tags_string,)
                else:
                    continue
        return (True, core_tags_string,)

class BlendTwoImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fg": ("IMAGE",),
                "bg": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "blend_image"

    def blend_image(self, fg, bg):
        try:
            from PIL import Image
            import numpy as np
            import torchvision.transforms as transforms
        except ImportError:
            raise ImportError("Please install Pillow to use node 'BlendImage'.")

        print(type(fg), type(bg))
        print(fg.shape, bg.shape) # (1, 512, 512, 4) (1, 512, 512, 4)
        # Convert to (1, 4, 512, 512)
        fg = fg.permute(0, 3, 1, 2)
        bg = bg.permute(0, 3, 1, 2)

        to_pil = transforms.ToPILImage()

        for (batch_number, image) in enumerate(fg):
            foreground_image = to_pil(image)

        for (batch_number, image) in enumerate(bg):
            background_image = to_pil(image)

        # Ensure both images are in RGBA format for alpha compositing
        foreground_image = foreground_image.convert("RGBA")
        background_image = background_image.convert("RGBA")

        # Calculate the position to paste the foreground image on the background
        bg_width, bg_height = background_image.size
        fg_width, fg_height = foreground_image.size
        offset_x = (bg_width - fg_width) // 2
        offset_y = (bg_height - fg_height) // 2

        # Create a new image to hold the blended result
        blended_image = Image.new("RGB", background_image.size)

        # Paste the background image onto the new image
        blended_image.paste(background_image, (0, 0))

        # Paste the foreground image onto the new image at the calculated position
        blended_image.paste(foreground_image, (offset_x, offset_y), foreground_image)

        # Convert the blended image back to tensor format
        to_tensor = transforms.ToTensor()
        blended_tensor = to_tensor(blended_image)

        # convert tensor to (1, c, h, w) format
        blended_tensor = blended_tensor.unsqueeze(0)
        # convert tensor to (1, h, w, c) format
        blended_tensor = blended_tensor.permute(0, 2, 3, 1)

        print(blended_tensor.shape)

        return (blended_tensor,)

class ListOperation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list1": ("LIST",),
                "list2": ("LIST",),
                "operation": (["list1 ∩ list2",
                               "list1 ∪ list2",
                               "list1 - list2",
                               "list2 - list1",
                               "list1 + list2 (in order)",
                               "list2 + list1 (in order)"],),
            }
        }

    RETURN_TYPES = ("LIST", "STRING",)
    RETURN_NAMES = ("LIST", "STRING",)
    FUNCTION = "list_operation"

    def list_operation(self, list1, list2, operation):
        # Cosidering some users use python < 3.10...
        new_list = []
        if operation == "list1 ∩ list2":
            for k in list1:
                if k in list2:
                    new_list.append(k)
        elif operation == "list1 ∪ list2":
            k_set = set()
            for k in list1:
                k_set.append(k)

            for k in list2:
                k_set.append(k)
            new_list = list(k_set)
        elif operation == "list1 - list2":
            for k in list1:
                if k not in list2:
                    new_list.append(k)
        elif operation == "list2 - list1":
            for k in list2:
                if k not in list1:
                    new_list.append(k)
        elif operation == "list1 + list2 (in order)":
            new_list = list1 + list2
        elif operation == "list2 + list1 (in order)":
            new_list = list2 + list1

        return (new_list, ", ".join(new_list))

class RemoveSaturation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "remove_saturation"

    def remove_saturation(self, image):
        try:
            from PIL import Image
            import numpy as np
            import torchvision.transforms as transforms
        except ImportError:
            raise ImportError("Please install Pillow to use node 'RemoveSaturation'.")
        image = image.permute(0, 3, 1, 2)

        to_pil = transforms.ToPILImage()

        for (batch_number, image) in enumerate(image):
            foreground_image = to_pil(image)
        foreground_image = foreground_image.convert('L')
        foreground_image = foreground_image.convert('RGB')

        to_tensor = transforms.ToTensor()
        blended_tensor = to_tensor(foreground_image)

        # convert tensor to (1, c, h, w) format
        blended_tensor = blended_tensor.unsqueeze(0)
        # convert tensor to (1, h, w, c) format
        blended_tensor = blended_tensor.permute(0, 2, 3, 1)

        print(blended_tensor.shape)
        return (blended_tensor,)
