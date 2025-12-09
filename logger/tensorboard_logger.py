

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
from PIL import Image
import numpy as np

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        #from datetime import datetime
        #now = datetime.now()
        #log_dir = log_dir + now.strftime("%Y%m%d-%H%M%S")
        # self.writer = tf.compat.v1.summary.FileWriter(log_dir)
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        # summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        # self.writer.add_summary(summary, step)
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
        self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        '''
        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            # scipy.misc.toimage(img).save(s, format="png")
            Image.fromarray((img * 255).astype(np.uint8)).save(s, format="PNG")

            # Create an Image object
            img_sum = tf.compat.v1.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])

            img_summaries.append(tf.compat.v1.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        '''
        images = np.array(images)
        if images.ndim == 3:  # (H, W, C)
            images = np.expand_dims(images, 0)  # (1, H, W, C)
        elif images.ndim == 2:  # (H, W)
            images = np.expand_dims(images, (0, -1))  # (1, H, W, 1)

        # Convert to float32 [0,1] range
        images = images.astype(np.float32)
        if images.max() > 1.0:
            images /= 255.0

        with self.writer.as_default():
            tf.summary.image(tag, images, step=step)
            self.writer.flush()
