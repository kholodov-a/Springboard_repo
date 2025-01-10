# Problem Statement Worksheet (Hypothesis Formation)

Develop an image classification model to automatically identify the six types of recyclable and non-recyclable materials from images of garbage. This solution aims to increase sorting accuracy and reduce labor costs in waste management facilities by 20% within six months of deployment.

### Context

Effective waste management is essential for sustainability. Automated garbage classification can enhance efficiency, cut costs, and boost recycling rates, directly aiding waste management facilities.

### Constraints within solution space

- Limited computational resources for training large-scale deep learning models.  
- Possible imbalances in the TrashNet dataset, leading to challenges in achieving high classification accuracy across all categories.  
- The TrashNet dataset is not sufficiently large, which may hinder the model’s ability to generalize well to new, unseen data.  

### Criteria for success

-	Achieve an image classification model with an accuracy of at least 90% on the TrashNet dataset.  
-	Ensure the model’s precision and recall for each waste category (e.g., paper, plastic, metal, glass, cardboard and trash) exceed 85%.  
-	Successfully deploy the model in a production environment with potetial for real-time sorting capabilities.  

### Stakeholders to provide key insight
__Stakeholders:__ Decision-makers of waste management companies, environmental agencies and NGOs, and city councils interested in recycling solutions.  
__Data Sources:__ The TrashNet dataset containing labeled images of various waste materials.

### Scope of solution space 
- Focus exclusively on classifying garbage into five categories: paper, plastic, metal, glass, cardboard and trash.
-	Ensure the solution has the capability to learn and be trained in parallel with its deployment in production, allowing for continuous improvement as new data becomes available.

### Key data sources 

The TrashNet dataset for training and evaluating the model. Currently, the dataset consists of 2527 images:
501 glass, 594 paper, 403 cardboard, 482 plastic, 410 metal and 137 trash

