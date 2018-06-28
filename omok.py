import numpy as np
import matplotlib.pyplot as plt
import copy

class Omok:
    # --------------------------------
    # 초기화
    # --------------------------------
    def __init__(self, gridSize, show_game= False, STONE_PLAYER1 = 1,STONE_PLAYER2 = -1
                 ,handicap_Reward = -1, win_Reward = 5,STONE_NONE = 0,STONE_MAX = 5):
        self.gridSize = gridSize
        self.nbStates = self.gridSize * self.gridSize #오목판의 크기
        self.state = np.zeros(self.nbStates, dtype=np.int32)
        self.curPlayer = STONE_PLAYER1 #흑이 선수를 둔다.
        self.STONE_PLAYER1 = STONE_PLAYER1
        self.STONE_PLAYER2 = STONE_PLAYER2
        self.STONE_NONE = STONE_NONE
        self.STONE_MAX = STONE_MAX
        self.handicap_Reward = handicap_Reward
        self.win_Reward = win_Reward

        self.total_game = 0
        self.total_reward = [[0],[0]] #흑, 백의 총 보상
        self.current_reward = [[0],[0]] #흑 백의 현재 보상

        self.reset()
        if show_game:
            self.fig, self.axis = self._prepare_display()

    # --------------------------------
    # 리셋
    # --------------------------------
    def reset(self):
        self.total_game += 1
        self.state = np.zeros(self.nbStates)
        self.state[int(self.nbStates / 2)] = 1

    def _prepare_display(self):
        plt.ion()  # interactive 모드
        """게임을 화면에 보여주기 위해 matplotlib 으로 출력할 화면을 설정합니다. 화면은 grid 사이즈 만큼."""
        fig, axis = plt.subplots(figsize=(self.gridSize, self.gridSize))

        # 화면을 닫으면 프로그램을 종료합니다.
        fig.canvas.mpl_connect('close_event', exit)

        # 바둑판 색에 맞추어 색 변경
        axis.set_facecolor('xkcd:puce')  # 참고 색상표 : https://xkcd.com/color/rgb/

        # 화면에 출력할 축의 길이를 설정 [X축: -1 ~ 10],[Y축: -1 ~ 10]
        plt.axis((-1, self.gridSize, -1, self.gridSize))

        # 바둑판의 선을 그려주는 내용
        for y in range(0, self.gridSize):
            plt.axhline(y=y, color='k', linestyle='-')
            plt.axvline(x=y, color='k', linestyle='-')

        return fig, axis


    # --------------------------------
    # 현재 상태 구함
    # --------------------------------
    def getState(self):
        return np.reshape(self.state, (1, self.nbStates))


    def _draw_screen(self, player, act, gameOver):
        title = " Avg. Reward: %d black_Reward: %d white_Reward: %d Total Game: %d" % (
            np.sum(self.total_reward) / self.total_game,
            self.current_reward[0],self.current_reward[1],
            self.total_game)

        # 제목, 들어갈 내용: 에피소드, 리워드, 등등
        plt.title(title + "Tobig's 5 go")
        if (gameOver == False):
            if (player == self.STONE_PLAYER1):
                plt.plot([act // self.gridSize], [act % self.gridSize], 'ko', markersize=30)
            else:
                plt.plot([act // self.gridSize], [act % self.gridSize], 'wo', markersize=30)
        else:
            # 게임이 끝났다면 바툭판을 원상복귀
            plt.cla()  # 초기화
            self.axis.set_facecolor('xkcd:puce')  # 참고 색상표 : https://xkcd.com/color/rgb/
            plt.title("Game Over")
            for y in range(0, self.gridSize):
                plt.axhline(y=y, color='k', linestyle='-')
                plt.axvline(x=y, color='k', linestyle='-')
        plt.pause(1)

    # --------------------------------
    # render
    # --------------------------------
    def render(self):
        state = np.reshape(self.state, (self.gridSize, self.gridSize))
        return print(state)

    # --------------------------------
    # 매칭 검사
    # 다섯개 붙어 있으면 true, 아님 false
    # --------------------------------
    def CheckMatch(self, player):
        for y in range(self.gridSize):
            for x in range(self.gridSize):
                # --------------------------------
                # 오른쪽 검사
                # --------------------------------
                match = 0

                for i in range(self.STONE_MAX):
                    if (x + i >= self.gridSize):
                        break

                    if (self.state[y * self.gridSize + x + i] == player):
                        match += 1
                    else:
                        break

                    if (match >= self.STONE_MAX):
                        if (player == 1):
                            self.total_reward[0] = [np.sum([self.total_reward[0], self.current_reward[0]])]
                        else:
                            self.total_reward[1] = [np.sum([self.total_reward[1], self.current_reward[1]])]
                        return True

                # --------------------------------
                # 아래쪽 검사
                # --------------------------------
                match = 0

                for i in range(self.STONE_MAX):
                    if (y + i >= self.gridSize):
                        break

                    if (self.state[(y + i) * self.gridSize + x] == player):
                        match += 1
                    else:
                        break;

                    if (match >= self.STONE_MAX):
                        if (player == 1):
                            self.total_reward[0] = [np.sum([self.total_reward[0], self.current_reward[0]])]
                        else:
                            self.total_reward[1] = [np.sum([self.total_reward[1], self.current_reward[1]])]
                        return True

                # --------------------------------
                # 오른쪽 대각선 검사
                # --------------------------------
                match = 0

                for i in range(self.STONE_MAX):
                    if ((x + i >= self.gridSize) or (y + i >= self.gridSize)):
                        break

                    if (self.state[(y + i) * self.gridSize + x + i] == player):
                        match += 1
                    else:
                        break

                    if (match >= self.STONE_MAX):
                        if (player == 1):
                            self.total_reward[0] = [np.sum([self.total_reward[0], self.current_reward[0]])]
                        else:
                            self.total_reward[1] = [np.sum([self.total_reward[1], self.current_reward[1]])]
                        return True

                # --------------------------------
                # 왼쪽 대각선 검사
                # --------------------------------
                match = 0

                for i in range(self.STONE_MAX):
                    if ((x - i < 0) or (y + i >= self.gridSize)):
                        break

                    if (self.state[(y + i) * self.gridSize + x - i] == player):
                        match += 1
                    else:
                        break

                    if (match >= self.STONE_MAX):
                        if (player == 1):
                            self.total_reward[0] = [np.sum([self.total_reward[0], self.current_reward[0]])]
                        else:
                            self.total_reward[1] = [np.sum([self.total_reward[1], self.current_reward[1]])]
                        return True

        return False

    # --------------------------------
    # 게임오버 검사
    # 리워드를 리턴
    # 이긴 player : 1, 진 player  : -1, 무승부 : 0
    # --------------------------------
    def getReward(self, player):
        if (self.CheckMatch(self.STONE_PLAYER1) == True):
            if (player == self.STONE_PLAYER1):
                return self.win_Reward
            else:
                return -self.win_Reward
        elif (self.CheckMatch(self.STONE_PLAYER2) == True):
            if (player == self.STONE_PLAYER1):
                return -self.win_Reward
            else:
                return self.win_Reward
        else:
            for i in range(self.nbStates):
                if (self.state[i] == self.STONE_NONE):
                    return 0
            return 0

    # 양수만 반환하는 리스트
    # filter 함수 사용시
    def positive(self, x):
        return x > 0

    def max_value(self, x):
        return x < self.nbStates

    def right_diag_up(self, act):
        idx = []
        temp_act = act
        for i in range(self.count_right(act)):
            idx.append(temp_act - (self.gridSize - 1))  # 오른 대각선 위
            temp_act = temp_act - (self.gridSize - 1)
        idx = list(filter(self.positive, list(idx)))
        idx = list(filter(self.max_value, list(idx)))
        return idx

    def right_diag_down(self, act):
        idx = []
        temp_act = act
        for i in range(self.count_left(act)):
            idx.append(temp_act + (self.gridSize - 1))  # 왼 대각선 아래
            temp_act = temp_act + (self.gridSize - 1)
        idx = list(filter(self.positive, list(idx)))
        idx = list(filter(self.max_value, list(idx)))
        return idx

    def right_diag(self, act):
        idx = self.right_diag_up(act) + self.right_diag_down(act)
        idx.append(act)
        idx.sort()
        return idx

    def left_diag_up(self, act):
        idx = []
        temp_act = act
        for i in range(self.count_left(act)):
            idx.append(temp_act - (self.gridSize + 1))  # 왼 대각선 위
            temp_act = temp_act - (self.gridSize + 1)
        idx = list(filter(self.positive, list(idx)))
        idx = list(filter(self.max_value, list(idx)))
        return idx

    def left_diag_down(self, act):
        idx = []
        temp_act = act
        for i in range(self.count_right(act)):
            idx.append(temp_act + (self.gridSize + 1))  # 오른 대각선 아래
            temp_act = temp_act + (self.gridSize + 1)
        idx = list(filter(self.positive, list(idx)))
        idx = list(filter(self.max_value, list(idx)))
        return idx

    def left_diag(self, act):
        idx = self.left_diag_up(act) + self.left_diag_down(act)
        idx.append(act)
        idx.sort()
        return idx

    def height(self, act):
        idx = []
        temp = act % self.gridSize
        for i in range(self.nbStates):
            if i % self.gridSize == temp:
                idx.append(i)
            else:
                pass
        idx.sort()
        return idx

    def width(self, act):
        idx = []
        temp = int(act / self.gridSize)
        for i in range(self.nbStates):
            if int(i / self.gridSize) == temp:
                idx.append(i)
            else:
                pass
        idx.sort()
        return idx

    # 오른쪽, 왼쪽으로 몇 칸이 있는지.
    def count_right(self, act):
        temp_act = act
        right_wall = list(range(self.gridSize - 1, self.nbStates, self.gridSize))  # 오른 벽

        while (not (temp_act in right_wall)):
            temp_act += 1
        return temp_act - act

    def count_left(self, act):
        temp_act = act
        left_wall = list(range(0, self.nbStates, self.gridSize))  # 왼 벽

        while (not (temp_act in left_wall)):
            temp_act -= 1
        return act - temp_act

    def is_seq(self,next_state, id, k):
        length = len(id) - k + 1

        seq = []
        for i in range(length):
            seq.append(list(range(i, i + k)))

        sequence = []
        for _, seq_ in enumerate(seq):
            sequence.append([id[i] for i in seq_])

        result = []
        for _, sequence_ in enumerate(sequence):
            result.append(np.abs(np.sum(next_state[sequence_])) == k)

        if np.sum(result) >= 1:
            return True
        else:
            return False

    def grid_check(self, act, player):
        next_state = copy.deepcopy(self.state)
        next_state[act] = player
        idx = [self.right_diag(act)  # 오른쪽 대각선의 좌표
            , self.left_diag(act)  # 왼쪽 대각선의 좌표
            , self.width(act), self.height(act)]  # 가로, 세로 좌표

        result = [[], [], []]
        # 1번째 list: act을 기준으로  오른대각, 왼대각, 가로, 세로를 탐색했을때, 놓여진 돌의 갯수
        # 2번째 list: 3개가 연속해서 놓여 있는지
        # 3번째 list: 4개가 연속해서 놓여 있는지

        for i, id in enumerate(idx):
            result[0].append(np.sum(next_state[id]))
            result[1].append(self.is_seq(next_state,id, 3))
            result[2].append(self.is_seq(next_state,id, 4))

        if np.sum([x == 1 for x in result[0]]) == 4:  # 전혀 엉뚱한데 두면 -1
            return -1
        elif np.sum(result[2]) >= 1:  # 4개가 연속으로 놓여 있는가?
            return 4

        elif np.sum(result[1]) >= 1:  # 3개가 연속으로 놓여 있는가?
            return 3
        elif np.sum([x == 4 for x in result[0]]) >= 1:  # 4개가 어느줄이든 걸쳐 있는가?
            return 2
        elif np.sum([x == 3 for x in result[0]]) >= 1:  # 3개가 어느 줄이든 걸쳐 있는가?
            return 1
        else:
            return 0

    # --------------------------------
    # action취해서 state update
    # player : 다음 player로 전환
    # 이미 놓은 자리에 또 놓으면 음수 리워드
    # --------------------------------
    def _update_stone(self, player, action):
        if (self.state[action] == 0.0):
            reward = 0
            self.state[action] = player  # 돌을 둔다.
        else:
            reward = self.handicap_Reward
            #랜덤으로 돌을 둔다.
            random_action = np.random.choice(np.where(env.state == 0.0)[0], 1)[0]
            self.state[random_action] = player  # 돌을 둔다.

        return self.state, -player, reward

    # --------------------------------
    # curPlayer에 맞는 state
    # --------------------------------
    def getCanonicalForm(self, state, curPlayer):
        return state * curPlayer

    # --------------------------------
    # 비어있는 곳에만 다음 수를 놓아야 하기 때문에 vacant인 곳에 1, 수가 놓여져 있는 곳에 0
    # --------------------------------
    def getVacant(self, state):
        vacant = np.zeros(self.nbStates)
        for i in range(self.nbStates):
            if (state[i] == 0):
                vacant[i] = 1
        return vacant

    # --------------------------------
    # 한 에피소드 동안 실행(게임이 끝날 때까지)
    # --------------------------------
    def step(self, action, player):
        # _update_stone()을 하기 전, 착수한 돌의 리워드를 반환
        stone_reward = self.grid_check(action, player)
        #돌을 실제로 착수
        next_state, next_player, handicap_reward = self._update_stone(player, action)
        #게임이 끝났는지 검사
        done = self.CheckMatch(player)
        #승패 리워드
        win_loose_reward = self.getReward(player)

        reward = stone_reward + win_loose_reward + handicap_reward
        if (player == 1):
            self.total_reward[0] = [np.sum([self.total_reward[0], [reward]])]
        else:
            self.total_reward[1] = [np.sum([self.total_reward[1], [reward]])]


        return next_state, reward, done, next_player

if __name__ == '__main__':
    env = Omok(7)

    env.step(action = 23, player= 1)
    env.step(action = 25, player= 1)
    env.step(action = 22, player= 1)

    env.render()
    print(env.total_reward)

    env.step(action = 0, player= -1)
    env.step(action = 1, player= -1)
    env.step(action = 2, player=-1)
    env.step(action=3, player=-1)
    env.step(action=4, player=-1)
    print(env.total_reward)

    env.render()
