import numpy as np
import copy
gridSize = 7


gridSize = gridSize
nbStates = gridSize * gridSize #절대 지우면 안됨.
state = np.zeros(nbStates, dtype=np.int32)

for i in range(nbStates):
    state[i] = i
state.reshape((gridSize, gridSize))

'''
세로 검사: 착수 위치% gridSize = k 일 때 (나머지가 같은 줄이 세로 줄)
    3%7
    24%7
    45%7
가로 검사: 착수 위치//gridSize = n 일 때, n부터 n + gridSize까지 검사
    int(21/7)
    int(24/7)
    int(27/7)
오른 대각선 : 착수위치 +- (gridSize -1)
24 + (gridSize-1)
왼 대각선 : 착수위치 +- (gridSize +1)
'''


# 양수만 반환하는 리스트
# filter 함수 사용시
def positive(x):
    return x > 0
def max_value(x):
    return x < nbStates


def right_diag_up(act):
    idx = []
    temp_act = act
    for i in range(count_right(act)):
        idx.append(temp_act - (gridSize - 1))  # 오른 대각선 위
        temp_act = temp_act - (gridSize - 1)
    idx = list(filter(positive, list(idx)))
    idx = list(filter(max_value, list(idx)))
    return idx
def right_diag_down(act):
    idx = []
    temp_act = act
    for i in range(count_left(act)):
        idx.append(temp_act + (gridSize - 1))  # 왼 대각선 아래
        temp_act = temp_act + (gridSize - 1)
    idx = list(filter(positive, list(idx)))
    idx = list(filter(max_value, list(idx)))
    return idx
def right_diag(act):
    idx = right_diag_up(act) + right_diag_down(act)
    idx.append(act)
    idx.sort()
    return idx



def left_diag_up(act):
    idx = []
    temp_act = act
    for i in range(count_left(act)):
        idx.append(temp_act - (gridSize + 1))  # 왼 대각선 위
        temp_act = temp_act - (gridSize + 1)
    idx = list(filter(positive, list(idx)))
    idx = list(filter(max_value, list(idx)))
    return idx
def left_diag_down(act):
    idx = []
    temp_act = act
    for i in range(count_right(act)):
        idx.append(temp_act + (gridSize + 1))  # 오른 대각선 아래
        temp_act = temp_act + (gridSize + 1)
    idx = list(filter(positive, list(idx)))
    idx = list(filter(max_value, list(idx)))
    return idx
def left_diag(act):
    idx = left_diag_up(act) + left_diag_down(act)
    idx.append(act)
    idx.sort()
    return idx


def height(act):
    idx = []
    temp = act % gridSize
    for i in range(nbStates):
        if i % gridSize == temp:
            idx.append(i)
        else:
            pass
    idx.sort()
    return idx

def width(act):
    idx = []
    temp = int(act / gridSize)
    for i in range(nbStates):
        if int(i / gridSize) == temp:
            idx.append(i)
        else:
            pass
    idx.sort()
    return idx


# 오른쪽, 왼쪽으로 몇 칸이 있는지.
def count_right(act):
    temp_act = act
    right_wall = list(range(gridSize - 1, nbStates, gridSize))  # 오른 벽

    while (not (temp_act in right_wall)):
        temp_act += 1
    return temp_act - act


def count_left(act):
    temp_act = act
    left_wall = list(range(0, nbStates, gridSize))  # 왼 벽

    while (not (temp_act in left_wall)):
        temp_act -= 1
    return act - temp_act


'''
사용 예시
'''
act = 10

right_diag_up(act)
right_diag_down(act)
left_diag_up(act)
left_diag_down(act)
height(act)
width(act)

'''
grid_check: 이전 state와 현재 next_state의 관계_(1) 를 파악해 
, 3개이상의 돌이 두워져 있다면, 1(reward)를 return 

(1)관계
    ->가로, 세로, 대각


'''

'''
state = np.zeros(nbStates, dtype=np.int32)

state[24] = 1
state[25] = 1

act = 26
'''
def is_seq(next_state,id, k):
    length = len(id) - k +1

    seq = []
    for i in range(length):
        seq.append(list(range(i,i+k)))

    sequence = []
    for _, seq_ in enumerate(seq):
        sequence.append([id[i] for i in seq_])

    result = []
    for _,sequence_ in enumerate(sequence):
        result.append(np.abs(np.sum(next_state[sequence_])) == k)

    if np.sum(result)>=1:
        return True
    else:
        return False

def grid_check(act, state,player):
    next_state = copy.deepcopy(state)  # Class에 넣을때는 함수 getNextState()로 수정하기
    next_state[act] = player
    idx = [right_diag(act)  # 오른쪽 대각선의 좌표
        , left_diag(act)  # 왼쪽 대각선의 좌표
        , width(act), height(act)]  # 가로, 세로 좌표

    result = [[],[],[]]
    # 1번째 list: act을 기준으로  오른대각, 왼대각, 가로, 세로를 탐색했을때, 놓여진 돌의 갯수
    # 2번째 list: 3개가 연속해서 놓여 있는지
    # 3번째 list: 4개가 연속해서 놓여 있는지

    for i, id in enumerate(idx):
        result[0].append(np.sum(next_state[id]))
        result[1].append(is_seq(next_state,id, 3))
        result[2].append(is_seq(next_state,id, 4))

    if np.sum([x == 1 for x in result[0]]) == 4: #전혀 엉뚱한데 두면 -1
        return -1

    elif np.sum(result[2])>=1: #4개가 연속으로 놓여 있는가?
        return 4
    elif np.sum(result[1])>=1: #3개가 연속으로 놓여 있는가?
        return 3
    elif np.sum([x == 4 for x in result[0]]) >= 1: #4개가 어느줄이든 걸쳐 있는가?
        return 2
    elif np.sum([x == 3 for x in result[0]]) >= 1: #3개가 어느 줄이든 걸쳐 있는가?
        return 1

    else:
        return 0

state = np.zeros(nbStates, dtype=np.int32)
state[0]=-1
state[1]=-1
state[2]=-1

grid_check(act, state,-1)