(ns com.benfle.machine-learning.concept
  "Concept Learning from Chapter 2 of Tom Mitchell's \"Machine Learning\"."
  (:require [clojure.set :as set]))

(def training-examples
  [[:sunny :warm :normal :strong :warm :same   true ]
   [:sunny :warm :high   :strong :warm :same   true ]
   [:rainy :cold :high   :strong :warm :change false]
   [:sunny :warm :high   :strong :cool :change true ]])

(defn values
  "The set of possible values at `idx`."
  [examples idx]
  (set (map #(nth % idx) examples)))

;; An hypothesis is represented as a vector of constraints:
;;  ?        Any value is acceptable
;;  ∅        No value is acceptable
;;  <value>  Specifc value is required

(defn match?
  "Whether the value matches the constraint."
  [constraint value]
  (case constraint
    ∅ false
    ? true
    (= constraint value)))

(defn positive?
  "Whether the hypothesis classifies the instance as positive."
  [hypothesis instance]
  (every? identity
          (map match?
               hypothesis
               instance)))

;; General-to-Specific Ordering of Hypthesis
;; Note: This is a partial order.

(defn most-general-hypothesis  [example] (vec (repeat (dec (count example)) '?)))
(defn most-specific-hypothesis [example] (vec (repeat (dec (count example)) '∅)))

(defn more-general?
  "Whether the hypothesis `h1` is more general than the hypothesis `h2`."
  [h1 h2]
  (every? identity
          (map #(or (= %1 '?)
                    (= %1 %2))
               h1
               h2)))

;; FIND-S: Finding a maximally specific hypothesis

(defn generalize
  "Generalize the hypothesis to be consistent with the example."
  [hypothesis example]
  (mapv #(cond
           (= %1 '?) '?
           (= %1 '∅) %2
           (= %1 %2) %1
           :else     '?)
        hypothesis
        example))

(defn find-s
  "The maximally specific hypothesis that is consistent will all positive examples."
  [examples]
  (loop [[example & tail] examples
         hypothesis (most-specific-hypothesis (first examples))]
    (tap> {:example example
           :hypothesis hypothesis})
    (if-not example
      hypothesis
      (recur tail
             (if-not (last example)
               hypothesis
               (generalize hypothesis example))))))

;; CANDIDATE-ELIMINATION ALGORITHM

(defn consistent?
  "Whether the hypothesis is consistent with the training example."
  [hypothesis example]
  (= (positive? hypothesis (butlast example))
     (last example)))

(defn minimal-specializations
  "Minimally specialize an hypothesis to cover a negative example."
  [hypothesis examples example]
  (->> hypothesis
       (map-indexed (fn [idx expect]
                      (when (= expect '?)
                        (mapv #(assoc hypothesis idx %)
                              (disj (values examples idx)
                                    (nth example idx))))))
       (apply concat)))

(defn candidate-elimination
  "The version space for this set of training examples.

  The version space is represented as a map with the following keys:
  :G the set of maximally general  hypotheses consistent with the examples
  :S the set of maximally specific hypotheses consistent with the examples"
  [examples]
  (loop [[example & tail] examples
         {:keys [G S] :as vs} {:G #{(most-general-hypothesis  example)}
                               :S #{(most-specific-hypothesis example)}}]
    (tap> {:G G :S S})
    (if-not example
      vs
      (recur tail
             (let [inconsistent? #(not (consistent? % example))]
               (if (last example)
                 ;; positive example
                 (let [G (set (remove inconsistent? G))
                       S (set/union (set (remove inconsistent? S))
                                    (->> (filter inconsistent? S)
                                         (map #(generalize % example))
                                         (remove inconsistent?)
                                         (filter (fn [h] (some #(more-general? % h) G)))
                                         set))
                       S (set (remove (fn [s] (some #(and (not (= s %)) (more-general? s %)) S)) S))]
                   {:G G :S S})
                 ;; negative example
                 (let [S (set (remove inconsistent? S))
                       G (set/union (set (remove inconsistent? G))
                                    (->> (filter inconsistent? G)
                                         (mapcat #(minimal-specializations % examples example))
                                         (remove inconsistent?)
                                         (filter (fn [h] (some #(more-general? h %) S)))
                                         set))
                       G (set (remove (fn [g] (some #(and (not (= % g)) (more-general? % g)) G)) G))]
                   {:G G :S S})))))))

(comment

  (require '[com.benfle.machine-learning.concept :as concept] :reload)

  (add-tap println)

  ;; FIND-S
  (def maximally-specific-hypothesis (concept/find-s concept/training-examples))
  (println maximally-specific-hypothesis)

  ;; CANDIDATE-ELIMINATION
  (def version-space (concept/candidate-elimination concept/training-examples))
  (println version-space)

  ;; Version Space for http://www2.cs.uregina.ca/~dbd/cs831/notes/ml/vspace/vs_prob1.html
  (def version-space (concept/candidate-elimination
                      [[:japan :honda    :blue  1980 :economy true]
                       [:japan :toyota   :green 1970 :sports  false]
                       [:japan :toyota   :blue  1990 :economy true]
                       [:usa   :chrysler :red   1980 :economy false]
                       [:japan :honda    :white 1980 :economy true]]))
  (println version-space)

  )
