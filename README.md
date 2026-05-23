STARTING- I starded by changing the values of my bot body with the help of gemini too that what is the best possible value of number of legs for me then changing each parameters one by one to compare which parameters made it better and which not but I didn't have all my codes as I was focusing on increasing my length of walk till gen 100 and didn't realised I should have collected data for each of my changes and thus saved only a few of my codes.


CHANGING BOT'S BODY PARAMETERS- changing it didn't showed me good results so I decided changing fitness paramters but I have tried to change shin length and thigh length one by one it didn't gave me good results and I also used gemini to tell me what values should I choose to maximise my stability and speed by asking it to derive the values through equations and constraints then after changing differnet parameters I also tried copying my friend's parameters but it just decreased he best walk so I thought I should focus on other paramters.


CHANGING FITNESS PARAMTERS- 
1. changed the penalty given for number of legs active by reducing the total score by multiplying with a factor of (total no. of legs active/4)
2. penalized the fitness score by -50 if all legs are not active........BEST distance till 100 gen was 24.31 if only this parameter is added
3. penalize excessive or wasteful joint movement i.e to reward the robot for distance, but fine it a small amount of "tax" for every bit of joint rotation it uses. This forces the neural network to find the smoothest, most energy-efficient quadruped gait possible code I added to make this required changes -
if "total_movement" in data:
# Scale it with a small fraction (e.g., 0.01) so it doesn't overpower distance                                                                                                                                   
energy_penalty = data["total_movement"] * 0.01                                                                                                                                                                   
score -= energy_penalty                                                                                                                                                                                   
4. activating custom trackers of my metrices into fitness function- code I used-                                                                                                                                    
score -= data.get("backward_frames", 0) * 0.02                                                                                                                                                                   
score -= data.get("air_frames", 0) * 0.01                                                                                                                                                                    
BEST DISTANCE AFTER INCLUDING ALL THESE CHANGES 26.27                                                                                                                                    


CHANGING MY-METRICES- 1. I thought it would have been better if my bot uses all legs equally as it will stabalize it more and thus should increase the speed so I changed the code which will Track Individual Foot Contacts in my metrices i.e                                                                                                                                                                                 
def my_metrics(step_data):                                                                                                                                                                                       
    # Let's track how often the robot limps on exactly 1 or 3 legs asymmetricially                                                                                                                                   
    # versus a clean 2-foot trot or 4-foot plant.                                                                                                                                                                    
    is_asymmetric = 1 if step_data["feet_down"] in [1, 3] else 0                                                                                                                                                     
    return {                                                                                                                                                                                                         
        # Frames where the creature moves backward.                                                                                                                                                                      
        "backward_frames": 1 if step_data["vx"] < -2 else 0,                                                                                                                                                             
        # Frames where no feet are on the ground at all.                                                                                                                                                                 
        "air_frames": 1 if step_data["feet_down"] == 0 else 0,                                                                                                                                                           
        # New metric tracking uneven weight distribution frames                                                                                                                                                          
        "asymmetric_frames": is_asymmetric                                                                                                                                                                               
        }                                                                                                                                                                                                     
and then i added penalty in fitness function i.e score -= data.get("asymmetric_frames", 0) * 0.05 and score += data["smoothness"] * 15.0  # High reward for a steady, perfectly even rhythm                


ENDING- after all the alterations tried the best codes of the team where run multiple time by all teamates in different generations sunc as 200 300 400 to check stability of results and to filter out the unstable codes to choose the best one.



The two codes I have attached have included all my changes as I have finally included all my changes into the last code i.e code2 which was consistent in walking till 22-26.5m.


















    
