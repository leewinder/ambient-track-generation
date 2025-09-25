[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)


# Ambient Track Generator
This project creates atmospheric videos by combining AI-generated images, upscaling them, adding subtle animations, and AI-generated music.  

Or at least it will do, one day - at the moment it just generates a nice 1024x1024, expands it to 1080p and the upscales it to 4k!

&nbsp; 
## A Caveat
I haven't run this on anything other than an M4 Mac Mini with 64GB of RAM, your milage may vary (if indeed you can make it move!)

&nbsp; 
## Installation

### Prerequisites 
- Python 3.11 (only tested with Python 3.11.10)
- Pyenv 2.6.7
- Hugging Face account
- Possibly a lot of other things, as it works on my machine and that's all I know

&nbsp; 
### Clone the repository

   ```
   git clone git@github.com:leewinder/ambient-track-generation.git
   cd ambient-track-generation
   ```

&nbsp; 
### Authentication set up

Create an authentication config in the root folder as `authentication.json`, and add your Hugging Face token (it only needs read access).  Easiest way to do this is to copy the `authentication_example.json` and rename it to `authentication.json` and add your token.  

When you first run the script, it might fail saying you need to accept the terms of the model being used.  Jump over to the URL it provides, accept the terms (if you want to) and run it again

&nbsp; 
### Create your generation file

Easiest way to do this is to copy the `generation_example.json` and rename it to `generation.json`.  

At this point, the only thing I would suggest changing in your file would be the `image_positive` and `image_negative` values, and possibly the `seed` value, which is currently fixed so the same prompt will generate the same image.  If you want it to change each time, remove that field entirely.

&nbsp; 
### Run the installation script

This will set up the various Python virtual environments and install packages etc. etc.  This will also download any additional models that the generation scripts needs, and put them in the ./models folder.
   ```
   bash ./install.sh
   ```


&nbsp; 
### Run the script

There's a couple of ways to run the script, the easiest being via the generate.sh script in the root folder
   ```
   bash ./generate.sh
   ```

You can also run the generate script directly, which lives in the `src\` folder, which will step through all the stages for you
   ```
   cd src/
   source "venv/bin/activate"
   python generate.py
   ```

In both the above cases, you can pass `--max-stages` (or `-m`) to either script, which will run that many stages and then successfully finish.  So if you pass 2, it'll run the image generation (stage 01), and image widening process (stage 02) and stop there.

#### Testing Individual Stages
If you want to run any of the stages independently you need to just jump into the relevant folder
```
   cd scripts/01\ -\ Generate\ Images/
   source "venv/bin/activate"
   python generate_image.py --output <the path to the root of this project e.g. /Users/lee/Documents/Development/ambient-track-generation>
```

&nbsp; 
## Output
The process will drop the results of the run into the `./output/` folder, with the date and time of the run.  

This will store the logs and environment used to run the generation step, plus the output files in the `./output/results` folder.  Since it's not finished yet, the output will be in the `./output/results/temp` folder, as it's simply the output of each step.


&nbsp; 
## Troubleshooting
### Common Issues
* It doesn't work
  - 🤷‍♂️

&nbsp; 
## Models Used
* Image generation: [Stable Diffusion XL](https://stablediffusionxl.com/)
* Out Painting: [Stable Diffusion XL](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
* Image Upscaling: [Real-ESRGAN x2+ (v0.2.1)](https://github.com/xinntao/Real-ESRGAN)

&nbsp; 
## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

- ✅ Free to use, copy, and modify for **non-commercial** purposes  
- ✅ Must provide attribution to the author  
- ❌ Commercial use is **not allowed**  

See the [LICENSE](./LICENSE) file for full details.
