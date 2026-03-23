from PIL import Image
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaption:
    def __init__(self):
        # Load the pre-trained BLIP image captioning model and processor
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def generate(self, img):
        if isinstance(img, str):
            # Open the image if the input is a file path
            img = Image.open(img)
        
        # Process the image input
        inputs = self.processor(images=img, return_tensors='pt')
        
        # Ensure the inputs are in tensor format (not a list)
        pixel_values = inputs.get('pixel_values')
        
        if pixel_values is None:
            raise ValueError("Processed inputs don't contain 'pixel_values'. Check the processor settings.")

        # Generate the caption using the model
        output = self.model.generate(pixel_values)

        # Decode the output and return the caption
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption

if __name__== '__main__':

    ic = ImageCaption()
  
    app = gr.Interface(
        fn = ic.generate,
        inputs = gr.Image(type = 'pil'),
        outputs = "text",
        description = "Upload an image to generate caption :)"
    )
    app.launch(share=True)