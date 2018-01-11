(ns com.benfle.machine-learning.tictactoe
  "Experiments with a learning system for the game of Tic-Tac-Toe."
  (:require [clojure.java.io :as io]
            [clojure.data.csv :as csv]))

;; A Tic-Tac-Toe board is represented by a vector of size 9.
;;
;; The mapping between the vector's indices and a space on the board:
;;   0 1 2
;;   3 4 5
;;   6 7 8
;;
;; We use:    to represent:
;;         0               an empty space
;;         1               a space taken by the current player
;;        -1               a space taken by the other player

(def empty-board
  [0 0 0 0 0 0 0 0 0])

;; Note that by using a representation of the board *relative* to
;; the current player, we respect the symmetry of the game and get
;; rid of some complexity. This requires, however, that we reverse
;; the board before each player's turn.

(defn reverse-board
  [board]
  (mapv - board))

;; A player can only play in an empty space

(defn successors
  "The possible next boards for the current user."
  [board]
  (->> (range 9)
       (filter #(zero? (nth board %)))
       (map #(assoc board % 1))))

;; Tic-Tac-Toe is won by the first player to get 3 in a row.
;; There are 8 possible rows.

(defn rows
  "Return the 8 rows of length 3."
  [board]
  (map #(mapv (partial nth board) %)
       [[0 1 2] [3 4 5] [6 7 8] ; 3 rows
        [0 3 6] [1 4 7] [2 5 8] ; 3 columns
        [0 4 8] [2 4 6]]))      ; 2 diagonals

(defn won?
  "Whether the current player has 3 in a row on the board."
  [board]
  (some #(apply = 1 %) (rows board)))

(defn lost?
  "Whether the other player has 3 in a row on the board."
  [board]
  (some #(apply = -1 %) (rows board)))

(defn done?
  "Whether one of the player won or the board is full."
  [board]
  (or (won? board)
      (lost? board)
      (every? (comp not zero?) board)))

;; A player is represented by a function that takes
;; a board and return it after having placed a `1`.

(defn random-player
  "Picks a move at random."
  [board]
  (rand-nth (successors board)))

;; A game is represented as a collection of at most 10 boards.

(defn game
  "Play a game against another player and return the boards."
  [player other]
  (loop [[board :as game] (list empty-board)
         [play & tail] (cycle [player #(-> %
                                           reverse-board
                                           other
                                           reverse-board)])]
    (let [game (conj game (play board))]
      (if (done? (first game))
        (reverse game)
        (recur game tail)))))

;; Learning framework

(defprotocol LearningSystem
  (experiment [system hypothesis]
    "Return a new experiment for the given hypothesis.")
  (perform [system experiment hypothesis]
    "Perform the experiment using the given hypothesis.")
  (critic [system performance hypothesis]
    "Analyze the performance and return a set of training examples.")
  (generalize [system hypothesis examples]
    "Generalize from the training examples and return a new hypothesis."))

(defn learn
  "Improve the hypothesis until there are no more  training examples."
  [system initial-hypothesis]
  (loop [hypothesis initial-hypothesis]
    (let [experiment (experiment system hypothesis)
          performance (perform system experiment hypothesis)]
      (if-let [examples (seq (critic system performance hypothesis))]
        (recur (generalize system hypothesis examples))
        hypothesis))))

;; Tic-Tac-Toe Learning System
;;
;; Task:
;;   Playing Tic-Tac-Toe
;; Performance Measure:
;;   Percent of games won
;; Training Experience:
;;   Games played against player with fixed evaluation function

(defprotocol TicTacToeHypothesis
  (evaluate [hypothesis board]
    "Evaluate the board with the current hypothesis.")
  (improve [hypothesis example]
    "Improve the hypothesis using the given example."))

(defn player-with-hypothesis
  "A player that uses the hypothesis to pick the best move."
  [hypothesis]
  (fn [board]
    (->> (successors board)
         (sort-by #(evaluate hypothesis %))
         last)))

(defn tic-tac-toe-learning-system
  "Return a learning system for Tic-Tac-Toe.

  The system learns the game by playing `n` games against `teacher`.

  The `report` function is called after each game and is passed
  the game to keep track of the performance of the learner."
  [teacher n report]
  (let [played (atom 0)]
    (reify LearningSystem
      (experiment [_ hypothesis]
        #(game % teacher))
      (perform [_ experiment hypothesis]
        (experiment (player-with-hypothesis hypothesis)))
      (critic [_ game hypothesis]
        (report game)
        (when-not (= n (swap! played inc))
          (into (mapv (fn [board successor]
                        [board (evaluate hypothesis successor)])
                      (butlast game)
                      (rest game))
                [[(last game)
                  (cond
                    (won? (last game))   100
                    (lost? (last game)) -100
                    :else                  0)]])))
      (generalize [_ hypothesis examples]
        (reduce (fn [hypothesis example]
                  (improve hypothesis example))
                hypothesis
                examples)))))

;; An hypothesis is represented as a linear combination of features.

(defrecord LinearCombination [features weights η]
  TicTacToeHypothesis
  (evaluate [_ board]
    (apply + (map * (map #(% board) features) weights)))
  (improve [hypothesis [board estimate]]
    (let [current (evaluate hypothesis board)
          weights (map #(+ %1 (* η
                                 (- estimate current)
                                 (%2 board)))
                       weights
                       features)]
      (assoc hypothesis :weights weights))))

(defn count-rows
  [board player-count other-player-count]
  (->> (rows board)
       (filter (fn [row]
                 (and (= player-count
                         (count (filter #(=  1 %) row)))
                      (= other-player-count
                         (count (filter #(= -1 %) row)))
                      (= (- 3 player-count other-player-count)
                         (count (filter #(=  0 %) row))))))
       count))

(def initial-hypothesis
  (map->LinearCombination
   {:features [(constantly 1)
               #(count-rows % 0 1)
               #(count-rows % 0 2)
               #(count-rows % 0 3)]
    :weights (repeat 0.5)
    :η 0.1}))

;; Keep statistics for a series of games

(defn new-stats-history
  []
  (atom (list {:games  0
               :wins   0
               :losses 0
               :ties   0})))

(defn update-stats
  [stats game]
  (let [new-stats (update stats :games inc)]
    (cond
      (won? (last game))  (update new-stats :wins   inc)
      (lost? (last game)) (update new-stats :losses inc)
      :else               (update new-stats :ties   inc))))

(defn reporter
  [stats-history]
  (fn [game]
    (swap! stats-history #(conj % (update-stats (first %) game)))))

(defn percent [total part] (if (zero? total) 0.0 (double (/ part total))))

(defn stats->csv
  [filename stats-history]
  (with-open [w (io/writer filename)]
    (csv/write-csv w (into [["wins" "losses" "ties"]]
                           (map (juxt #(percent (:games %) (:wins %))
                                      #(percent (:games %) (:losses %))
                                      #(percent (:games %) (:ties %)))
                                (reverse stats-history))))))

;; TODO:
;; [ ] Test learnt hypothesis against different styles of players
;; [ ] Create proper graphs, do not rely on CSV
;; [ ] Experiment with different features, hypothesis, teachers

(comment

  (require '[com.benfle.programming.learning.tictactoe] :reload)
  (in-ns 'com.benfle.programming.learning.tictactoe)

  (def stats-history (new-stats-history))
  (def n 1000)
  (def hypothesis initial-hypothesis)
  (def teacher (player-with-hypothesis hypothesis))
  ;; (def teacher random-player)
  (def system (tic-tac-toe-learning-system teacher
                                           n
                                           (reporter stats-history)))
  (def learnt-hypothesis (learn system hypothesis))
  (stats->csv "stats.csv" @stats-history)

  )
