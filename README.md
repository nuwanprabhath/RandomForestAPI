**API Documentation**
=====================
1. `/predictProbAll`
   Returns top 5 classes of highest probabilities for given attributes.
       ```
       Method Type: POST
       Accept: JSON
       Example: 
            Request: {"STATE": 3, "INTENTION": 2, "CAKE": 1, "VEGETABLE": 1, "MEAT": 1, "FISH": 5}
            Response: {
                        "PREDICTIONS": [
                          {
                            "PREDICTION": 5,
                            "PROBABILITY": 0.80000000000000004
                          },
                          {
                            "PREDICTION": 4,
                            "PROBABILITY": 0.20000000000000001
                          },
                          {
                            "PREDICTION": 1,
                            "PROBABILITY": 0
                          },
                          {
                            "PREDICTION": 2,
                            "PROBABILITY": 0
                          },
                          {
                            "PREDICTION": 3,
                            "PROBABILITY": 0
                          }
                        ]
                      }
       ```

2. `/predictProb`
   Returns max class with probability for that classification.
       ```
       Method Type: POST
       Accept: JSON
       Example: 
             Request: {"STATE": 3, "INTENTION": 2, "CAKE": 1, "VEGETABLE": 1, "MEAT": 1, "FISH": 5}
             Response: {
                         "PREDICTION": 5,
                         "PROBABILITY": 0.80000000000000004
                       }
       ```

3. `/predict`
    Returns only max class for given attributes.
    ```    
        Method Type: POST
        Accept: JSON
        Example: 
              Request: {"STATE": 3, "INTENTION": 2, "CAKE": 1, "VEGETABLE": 1, "MEAT": 1, "FISH": 5}
              Response: {
                         "PREDICTION": 5,
                        }
    ```  
4. `/train`
    Adds given attributes to CSV and regenerates decision tree.
    ```    
        Method Type: POST
        Accept: JSON
        Example: 
              Request: {"DESTINATION":1, "STATE": 3, "INTENTION": 2, "FISH":4}
              Response: "success"
    ```  
           
5. `/resetCsv`
    Remove all data from training CSV and adds only column DESTINATION 
    ```    
        Method Type: GET
        Example: 
              Response: "success"
    ```  
