(ns com.benfle.machine-learning.concept
  "From Chapter 2 of Tom M. Mitchell's \"Machine Learning\"."
  (:require [clojure.set :as set]))

;; Training example

(defrecord TrainingExample [data positive?])

(def training-examples
  (map #(map->TrainingExample
         {:data (zipmap [:sky :air-temp :humidity :wind :water :forecast]
                        (butlast %))
          :positive? (last %)})
       [[:sunny :warm :normal :strong :warm :same   true ]
        [:sunny :warm :high   :strong :warm :same   true ]
        [:rainy :cold :high   :strong :warm :change false]
        [:sunny :warm :high   :strong :cool :change true ]]))

;; An hypothesis is represented as a map mapping
;; and attribute to:
;;  ?        Any value is acceptable
;;  ∅        No value is acceptable
;;  <value>  Specifc value is required

(defn predict
  "Predict the result based on the hypothesis."
  [hypothesis data]
  (not
   (some (fn [[attr expect]]
           (or (= '∅ expect)
               (and (not= expect '?)
                    (not= expect (get data attr)))))
         hypothesis)))

(defn consistent?
  "Whether the hypothesis is consistent with the example."
  [hypothesis example]
  (= (predict hypothesis (:data example))
     (:positive? example)))

(defn generalize
  "Generalize the hypothesis to be consistent with the example."
  [hypothesis example]
  (->> hypothesis
       (map (fn [[attr value]]
              [attr (cond
                      (= value '?) value
                      (= value '∅) (get (:data example) attr)
                      (= value (get (:data example) attr)) value
                      :else '?)]))
       (into {})))

(defn minimal-specializations
  "Minimally specialize an hypothesis to cover a negative example."
  [hypothesis examples example]
  (mapcat (fn [[attr value]]
            (when (= value '?)
              (map #(assoc hypothesis attr %)
                   (disj (->> examples
                              (map #(get (:data %) attr))
                              set)
                         (get (:data example) attr)))))
          hypothesis))

(defn maximally-specific-hypothesis
  [example]
  (zipmap (keys (:data example))
          (repeat '∅)))

(defn maximally-general-hypothesis
  [example]
  (zipmap (keys (:data example))
          (repeat '?)))

(defn more-specific?
  "Whether h1 is strictly more specific to h2.

  Since  the relation only defines a *partial* order
  on the hypothesis space, (more-general? h1 h2)
  doesn't follow from (not (more-specific? h1 h2))."
  [h1 h2]
  (and (not= h1 h2)
       (every? #(or (= (h2 %) '?)
                    (= (h2 %) (h1 %)))
               (keys h1))))

(defn more-general?
  "Whether h1 is strictly more general to h2."
  [h1 h2]
  (and (not= h1 h2)
       (every? #(or (= (h1 %) '?)
                    (= (h1 %) (h2 %)))
               (keys h1))))

;; find-s algorithm

(defn find-s
  "The maximally specific hypothesis that is consistent will all positive examples."
  [examples]
  (loop [[example & tail] examples
         hypothesis (maximally-specific-hypothesis (first examples))]
    (println "example" example)
    (println "hypothesis" hypothesis)
    (if-not example
      hypothesis
      (recur tail
             (if-not (:positive? example)
               hypothesis
               (generalize hypothesis example))))))

(defn set-remove
  "Remove from the set the elements that do not match the given predicate."
  [pred s]
  (set (remove pred s)))

(defn trace [o] (println o) o)

(defn candidate-elimination
  "The version space for this set of training examples.

  The version space is represented as a map with the following keys:
  G the set of maximally general  hypotheses consistent with the examples
  S the set of maximally specific hypotheses consistent with the examples"
  [examples]
  (loop [[example & tail] examples
         {:keys [G S] :as vs} {:G #{(maximally-general-hypothesis example)}
                               :S #{(maximally-specific-hypothesis example)}}]
    (println "")
    (println "S" S)
    (println "G" G)
    (if-not example
      vs
      (let [not-consistent? #(not (consistent? % example))]
        (if (:positive? example)
          (let [G (set-remove not-consistent? G)]
            (recur tail
                   {:G G
                    :S (let [S (set/union
                                (set-remove not-consistent? S)
                                (->> (filter not-consistent? S)
                                     (map #(generalize % example))
                                     (filter #(consistent? % example))
                                     (filter (fn [s] (some #(more-general? % s) G)))
                                     set))]
                         (set-remove (fn [s] (some #(more-general? s %) S)) S))}))
          (let [S (set-remove not-consistent? S)]
            (recur tail
                   {:G (let [G (set/union
                                (set-remove not-consistent? S)
                                (->> (filter not-consistent? G)
                                     (mapcat #(minimal-specializations % examples example))
                                     (filter #(consistent? % example))
                                     (filter (fn [s] (some #(more-specific? % s) S)))
                                     set))]
                         (set-remove (fn [g] (some #(more-specific? g %) G)) G))
                    :S S})))))))

(comment

  (require '[com.benfle.machine-learning.concept] :reload)
  (in-ns 'com.benfle.machine-learning.concept)

  (def h (find-s training-examples))

  (def vs (candidate-elimination training-examples))

  )
