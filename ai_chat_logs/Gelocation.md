### User Input

Can you write the logic to check if particular region or state has high late_delivery_rate

*Viewed [observation.py](file:///c:/Users/Hunain%20Mulla/Desktop/Bismilla/ai_chat_logs/observation.py) *

*Edited relevant file*

### Planner Response

I have successfully added the logic to [observation.py](cci:7://file:///c:/Users/Hunain%20Mulla/Desktop/Bismilla/ai_chat_logs/observation.py:0:0-0:0)! 

The script will now do two things:
1. **Analyze delay vs review score:** It calculates the average review score for "On Time/Early" deliveries vs "Delayed" deliveries. It will also calculate the correlation to show exactly how much the delay severity impacts the review score.
2. **Find High Delay Regions**: It joins the `orders` dataset with the `customers` dataset to access the `customer_state`. It then groups all the orders by state to calculate the **Late Delivery Rate (%)**. Finally, it specifically targets regions with significant order volumes (>100 orders) and sorts them descending to see which states have the worst delays.

You can run the script using your terminal:
```bash
python observation.py
```

It should print a nice report showing the worst performing states as well as the impact delay has on the bad reviews. Let me know what interesting patterns emerge when you test this out!