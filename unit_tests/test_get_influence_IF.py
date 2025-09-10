import numpy as np
import pytest
from package.RankAMIP.logistic import LogisticAMIP
from package.RankAMIP.data_script import simulate_bt_design_matrix, make_BT_design_matrix
import pandas as pd


class TestGetInfluenceIF:
    """Test suite for get_influence_IF function"""
    
    def test_influence_halves_equal(self):
        """Test that influence[:actual_n] and influence[actual_n:] have equal magnitude"""
        np.random.seed(42)
        X, y = simulate_bt_design_matrix(num_teams=4, num_games=16, seed=42) # generate simulated data.
        
        # Create ties indicator vector that is 1 in 50% of spots where y is 0
        zeros_indices = np.where(y == 0)[0]
        num_ties = int(len(zeros_indices) * 0.5)
        tie_indices = np.random.choice(zeros_indices, size=num_ties, replace=False)
        ties = np.zeros_like(y)
        ties[tie_indices] = 1
        ties
        
        # Create weighted version by duplicating data
        X_weighted = np.concatenate((X, X))
        y_weighted = np.concatenate((y, y + ties))

        ### check on CBA data.
        # import pickle
        # with open('data/chatBotArena_wtd.pkl', 'rb') as f:
        #     X_weighted, y_weighted = pickle.load(f)
        
        model = LogisticAMIP(X_weighted, y_weighted, weighted=False)
        model_wtd = LogisticAMIP(X_weighted, y_weighted, weighted=True)
        # check that X_weighted and y_weighted are both even.
        print("number of rows in X_weighted is:", X_weighted.shape)
        print("number of rows in y_weighted is:", y_weighted.shape)
        
        for dim in range(model.__p__):
            influence = model.get_influence_IF(dim)
            influence_wtd = model_wtd.get_influence_IF(dim) # wted returns the sum of two halves.

            # check that the length of the vector of influence scores is even.
            print("length of influence unweighted is:", len(influence)) # treating as if 2 matrices were 1 long list.
            print("length of influence weighted is:", len(influence_wtd))

            actual_n = len(influence) // 2
            
            first_half = influence[:actual_n]
            second_half = influence[actual_n:]

            print("ties are:", ties[:20])
            print("first half is:", first_half[:20])
            print("second half is:", second_half[:20])
            print("first plus second is:", first_half + second_half)
            print("influence weighted is:", influence_wtd)

            # Check that halves are equal in magnitude---this is not necessarily the case for ties, because hat(p) is not the same when you flip the sign of a data point.
            first_half_norm = np.sum(first_half**2)
            second_half_norm = np.sum(second_half**2)

            print("first half norm is:", first_half_norm)
            print("second half norm is:", second_half_norm)

            # check that they are equal.
            np.testing.assert_allclose(first_half + second_half, influence_wtd, rtol=1e-10, 
                                        err_msg=f"first + second and influence weighted should be equal.")
        

    def test_edge_case_all_ties(self):
        """Test edge case with all ties (weighted data)"""
        data = {
            'team1': ['A', 'B', 'C', 'C', 'A', 'B'],
            'team2': ['B', 'C', 'A', 'B', 'B' , 'A'],
            'winner': [0, 0, 0, 0, 0, 0],  # All losses (will become ties)
            'tie': [1, 1, 1, 1, 1, 1]      # note, in the case when 1 element is not a tie (try changing the first one to be a non-tie) (suspect this happens for all odd number of non-ties), then the corresponding influence scores are not same magnitudes as their other halves.
        }
        df = pd.DataFrame(data)
        ties = df['tie']
        
        X_weighted, y_weighted, player_to_id = make_BT_design_matrix(df, weight_tie=True)
        
        model = LogisticAMIP(X_weighted, y_weighted, weighted=False)
        model_wtd = LogisticAMIP(X_weighted, y_weighted, weighted=True)
        # check that X_weighted and y_weighted are both even.
        print("number of rows in X_weighted is:", X_weighted.shape)
        print("number of rows in y_weighted is:", y_weighted.shape)
        
        for dim in range(model.__p__):
            influence = model.get_influence_IF(dim)
            influence_wtd = model_wtd.get_influence_IF(dim) # wted returns the sum of two halves.

            # check that the length of the vector of influence scores is even.
            print("length of influence unweighted is:", len(influence)) # treating as if 2 matrices were 1 long list.
            print("length of influence weighted is:", len(influence_wtd))

            actual_n = len(influence) // 2
            
            first_half = influence[:actual_n]
            second_half = influence[actual_n:]

            print("ties are:", ties[:20])
            print("first half is:", first_half[:20])
            print("second half is:", second_half[:20])
            print("first plus second is:", first_half + second_half)
            print("influence weighted is:", influence_wtd)

            # Check that halves are equal in magnitude---this is not necessarily the case, because hat(p) is not necessarily the same when you flip the sign of a data point.
            first_half_norm = np.sum(first_half**2)
            second_half_norm = np.sum(second_half**2)

            print("first half norm is:", first_half_norm)
            print("second half norm is:", second_half_norm)

            # check that they are equal.
            np.testing.assert_allclose(first_half + second_half + 1, influence_wtd, rtol=1e-10, 
                                        err_msg=f"first + second and influence weighted should be equal.")


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
