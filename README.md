# Tomer Keren`s Home Task for Saips

# Defects Detection in Silicon Wafers

### Installation and preparation
1. install all requirement packages for this project
```
pip install -r requirements.txt
```
### Usage

images directory tree:
```
├── images\
│    │
│    ├── case1ReferenceImage.jpg
|    ├── case1InspectedImage.jpg
|    ├── case2ReferenceImage.jpg
|    ├── case2InspectedImage.jpg
     ...
```  
```
python main.py 
optional arguments:
  -h, --help            show this help message and exit
  -s, --images-source   path to images directory
  --show-results        activate results figures
```
Usage example
```
python main.py --show-results
```

### Examples
<table>
  <tr> 
    <td>
         <tr>
<img src=examples/Case1.png    width="900" height="300">
       </tr>
       <tr>
         Finding SOME of the defects, while not finding others
         </tr>
       <tr>
<img src=examples/KP.png   width="600" height="300">
       </tr>
       <tr>  
       Example of aligning image by key points
</table>
