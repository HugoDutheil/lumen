import numpy as np
from pydantic import baseModel
from time import localtime

# TODO: create a baseModel for the data format to compare: pd.DataFrame(columns=["first_name", "first_percentage", "second_name", "second_percentage"])

# Should we give back the prediction + the value found for the true values? => better benchmarking for percentage and ranking
def MSE(y: np.array, y_true: np.array) -> float:
    return np.mean((y-y_true)**2)


def count_hallucinations(components_pred, components_true) -> tuple[int, int]:
    hallucinations_count = 0
    found = 0
    found_both, found_one, found_none = 0,0,0

    for i in range(len(components_true)):
        correct = 0
        if components_pred.first_name[0] in (components_true.first_name[i], components_true.second_name[i]):
            correct += 1

        if components_pred.first_name[0] in (components_true.first_name[i], components_true.second_name[i]):
            correct += 1
        
        if correct == 2:
            found_both += 1
        if correct == 1:
            found_one += 1
        if correct == 0:
            found_none += 1

        found += correct
        hallucinations_count += 2-correct

    return found, hallucinations_count, found_both, found_one, found_none

def calc_percentage_loss(
        components_pred: np.array,
        percentage_pred: np.ndarray, 
        components_true: np.array,
        percentage_true: np.ndarray, 
        lossFunc: Callable = MSE) -> float:

    y_pred = np.zeros(2*len(components_true))
    y_true = np.zeros(2*len(components_true))

    for i in range(len(components_true)):
        if components_pred.first_name[i] == components_true.first_name[i]:
            y_pred[2*i] = percentage_pred.first_percentage[i]
            y_true[2*i] = percentage_true.first_percentage[i]

        elif components_pred.second_name[i] == components_true.first_name[i]:
            y_pred[2*i] = percentage_pred.second_percentage[i]
            y_true[2*i] = percentage_true.first_percentage[i]
        else:
            y_pred[2*i] = 0
            y_true[2*i] = percentage_true.first_percentage[i]

        if components_pred.first_name[i] == components_true.second_name[i]:
            y_pred[2*i + 1] = percentage_pred.first_percentage[i]
            y_true[2*i + 1] = percentage_true.second_percentage[i]

        elif components_pred.second_name[i] == components_true.second_name[i]:
            y_pred[2*i + 1] = percentage_pred.second_percentage[i]
            y_true[2*i + 1] = percentage_true.second_percentage[i]
        else:
            y_pred[2*i + 1] = 0
            y_true[2*i + 1] = percentage_true.second_percentage[i]
        
    return lossFunc(y_pred/100, y_true/100), y_pred, y_true
    


def benchmark(pipeline: callable, args: list, kwargs: list[list], expected_results: np.ndarray) -> Callable:

    def run_pipeline(expected_results: np.ndarray) -> None:
        start = localtime()
        predictions, unprocessable = pipeline(**kwargs) #TODO: make silent ? > /dev/null
        end = localtime()

        assert len(prediction) == len(expected_results)

        components_true = expected_results.loc[["first_name", "second_name"]]
        components_pred = predictions.loc[["first_name", "second_name"]]
        percentage_true = expected_results.loc[["first_percentage", "second_percentage"]]
        percentage_pred = predictions.loc[["first_percentage", "second_percentage"]]

        found, hallucinations_count, found_both, found_one, found_none = count_hallucinations(components_pred, components_true)

        percentage_loss, y_pred, y_true = calc_percentage_loss(percentage_pred, percentage_true)
        
        relError = (y_true - y_pred)/y_true
        hist = np.histogram(relError, bins=20, density=True)
        mean, std = np.mean(hist), np.std(hist)

        
        print(f"========= BENCHMARK RESULTS =========")
        print(f"Total training time: {end-start}")
        print(f"Time per item: {(end-start)/len(prediction)}")
        print("-------------------------------------")
        print(f"Number of elements correctly identified: {found}")
        print(f"Number of elements hallucinated: {hallucinations_count}")
        print(f"Percentage of cases where x elemnts were found: \n
                                            both: {both_found}\n
                                            one : {one_found}\n
                                            none: {none_found}"
              )
        print("-------------------------------------")
        print(f"Loss on the predicted percentages: {percentage_loss}")
        print(f"Relative error distribution: \n
              Mean: {mean}\n
              std: {std}"
              )
        print("====================================")


    return run_pipeline
