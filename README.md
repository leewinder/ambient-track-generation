[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)


# Ambient Track Generator
This project creates atmospheric videos by combining AI-generated images, upscaling them, adding subtle animations, and AI-generated music.  Or at least it will do, one day - at the moment it just generates a nice 1024x1024 image!

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
### Run the installation script

This will set up the various Python virtual environments and install packages etc. etc.  You can do it manually if you want, but that's a hassle
   ```
   bash ./install.sh
   ```

&nbsp; 
### Authentication set up

Create an authentication config in the root folder as `authentication.json`, and add your Hugging Face token (it only needs read access)
   ```
   {
     "huggingface": <your hugging face token>
   }
   ```
When you first run the script, it might fail saying you need to accept the terms of the model being used.  Jump over to the URL it provides, accept the terms (if you want to) and run it again

&nbsp; 
### Create your config file

Easiest way to do this is to copy the `config_example.json` and rename it to `config.json`.  

At this point, the only thing I would suggest changing in your config file would be the `image_positive` and `image_negative` values, and possibly the `seed` value, which is currently fixed so the same prompt will generate the same image.  If you want it to change each time, remove that field entirely.

&nbsp; 
### Run the script

Since there's only one script at the moment...
   ```
   cd scripts/01\ -\ Generate\ Images/
   python generate_image.py --output <the path to the root of this project e.g. /Users/lee/Documents/Development/ambient-track-generation>
   ```
&nbsp; 
## Sample Output
Very little at the moment!  If you haven't changed anything in the config file other than the positive and negative prompts, it'll create an image in `/result/temp/01_initial_image.png`

&nbsp; 
## Troubleshooting
### Common Issues
* It doesn't work
  - 🤷‍♂️

&nbsp; 
## Models Used
Image generation: [Stable Diffusion XL](https://stablediffusionxl.com/)

&nbsp; 
## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

- ✅ Free to use, copy, and modify for **non-commercial** purposes  
- ✅ Must provide attribution to the author  
- ❌ Commercial use is **not allowed**  

See the [LICENSE](./LICENSE) file for full details.
