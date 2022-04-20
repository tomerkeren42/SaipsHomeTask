# Tomer Keren`s Home Task for Saips

# Defects Detection in Silicon Wafers

### Installation and preparation
1. install all requirement packages for this project
```
pip install -r requirements.txt
```
### Usage

images directory tree:
  images
    |
     caseNreference.jpg
     caseNinspected.jpg
     ...
     
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
<img src=<enter here image reference>    width="300" height="200">
<img src=<enter here image inspected>    width="300" height="200">
<img src=<enter here image results path> width="300" height="200">
    </td>
 </tr>
</table>
