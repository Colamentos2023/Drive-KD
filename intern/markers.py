img_start = "<img>"
img_end = "</img>"
user_start = "<|im_start|>user\n"
user_end = "<|im_end|>"
assistant_start = "<|im_start|>assistant\n"
assistant_end = "<|im_end|>"

image_marker = (img_start + "{}"+ img_end + "\n").format
user_marker = (user_start + "{}" + user_end + "\n").format
assistant_marker = (assistant_start + "{}" + assistant_end + "\n").format

image_pattern = rf"({img_start}.*?{img_end})"

user = user_marker
assistant = assistant_marker