# WILMFlo
Python package to derive both mining and processing energy cost of materials per prospective scenario and potential surplus and deficit of resources over time

## Potential uses of WILMFlo package

This package enable to calculate, on a yearly basis, the resource flows from extractio in use stocks, passing by fabrication of materials, then to collection and sorting, to recycling. 

The main WILMFlo 1.0 outputs are the following:

- Yearly extraction flows of 50 resources (e.g. copper, indium) and 3 fossil fuels (crude oil, coal, natural)
- Yearly fabrication of 72 materials (e.g. aluminium 1000 series alloy, copper)
- Yearly inflows to in-use stocks of 72 materials into 90 technologies
- Yearly outflows from in-use stocks to collection and sorting
- Yearly outflows from collection and sorting to recycling
- Yearly outflows from recycling to fabrication of materials
- Yearly dissipative flows at each life cycle stages
- Yearly surplus and deficit of resources at the extraction stage
- Yearly mining energy consumption for the 50 resources and 3 fossil fuels

A template to customize your scenario with your own data will be made available soon.

Results are available on Zenodo at :

## Dependencies

ODYM (https://github.com/IndEcol/ODYM)


## Author
Titouan Greffe (greffe.titouan@uqam.ca)

## Reference
Greffe, T., Margni, M., & Bulle, C. (2025). Evaluation of metals mining and processing energy consumption and potential metals supply–demand mismatch towards 2100. Resources, Conservation and Recycling, 223, 108482. https://doi.org/10.1016/j.resconrec.2025.108482 


