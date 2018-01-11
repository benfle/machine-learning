(ns com.benfle.machine-learning.rules
  "From Chapter 10 of Tom Mitchell's \"Machine Learning\".")

;; data

(defn play-tennis
  [vals]
  (zipmap [:outlook :temperature :humidity :wind :play-tennis?]
          vals))

(def examples
  (map #(play-tennis %)
       [[:sunny    :hot  :high   :weak   false]
        [:sunny    :hot  :high   :strong false]
        [:overcast :hot  :high   :weak   true]
        [:rain     :mild :high   :weak   true]
        [:rain     :cool :normal :weak   true]
        [:rain     :cool :normal :strong false]
        [:overcast :cool :normal :strong true]
        [:sunny    :mild :high   :weak   false]
        [:sunny    :cool :normal :weak   true]
        [:rain     :mild :normal :weak   true]
        [:sunny    :mild :normal :strong true]
        [:overcast :mild :high   :strong true]
        [:overcast :hot  :normal :weak   true]
        [:rain     :mild :high   :strong false]]))

;; sequential covering (propositional rules)

(defn entropy
  "The entropy of the examples relative to this target attribute."
  [examples target-attribute]
  (->> examples
       (map target-attribute)
       frequencies
       vals
       (map #(double (/ % (count examples))))
       (map #(- (* % (/ (Math/log %)
                        (Math/log 2)))))
       (reduce + 0)))

(defn match
  "Wether the hypothesis match the example."
  [hypothesis example]
  (every? (fn [[attr value]]
            (cond
              (= value '?) true
              (= value '∅) false
              :else (= value (get example attr))))
          hypothesis))

(defn performance
  "The performance of the hypothesis for these examples."
  [hypothesis examples target-attribute]
  (- (entropy (filter (partial match hypothesis)
                      examples)
              target-attribute)))

(defn most-frequent-value
  "The most frequent value of this attribute in the examples."
  [examples attribute]
  (->> examples
       (map attribute)
       frequencies
       (sort-by val)
       last
       first))

(defn specialize
  "Specialize the hypothesis using the given constraint.

  Return nil, if we could not generate a consistent hypothesis."
  [hypothesis [attr value]]
  (when (= (hypothesis attr) '?)
    (assoc hypothesis attr value)))

(defn- new-candidate-hypotheses
  [candidate-hypotheses examples target-attribute]
  (let [constraints (set (mapcat #(map (fn [v] [% (% v)]) examples)
                                 attributes))]
    (->> (for [h candidate-hypotheses
               c constraints]
           (specialize h c))
         (remove nil?)
         set
         (sort-by #(performance % examples target-attribute)))))

(defn learn-1
  "Learn one rule."
  [examples target-attribute attributes k]
  (let [performance #(performance % examples target-attribute)
        next-candidates #(new-candidate-hypotheses % examples target-attribute)]
    (loop [best-hypothesis (zipmap attributes (repeat '?))
           candidate-hypotheses #{best-hypothesis}]
      (if (empty? candidate-hypotheses)
        {:if best-hypothesis
         :then (most-frequent-value (filter (partial match best-hypothesis)
                                            examples)
                                    target-attribute)}
        (let [new-candidate-hypotheses (next-candidates candidate-hypotheses)]
          (recur (if (< (performance best-hypothesis)
                        (performance (last new-candidate-hypotheses)))
                   (last new-candidate-hypotheses)
                   best-hypothesis)
                 (take-last k new-candidate-hypotheses)))))))

(defn rule-performance
  [rule examples target-attribute]
  (let [matched-examples (filter #(match (:if rule) %)
                                 examples)]
    (if (every? #(not (target-attribute %)) examples)
      0
      (/ (count (filter #(= (:then rule) (target-attribute %))
                        matched-examples))
         (count matched-examples)))))

(defn learn
  [examples target-attribute attributes threshold]
  (loop [remaining-examples examples
         rules #{}]
    (let [rule (learn-1 remaining-examples target-attribute attributes 10)]
      (if (< (rule-performance rule remaining-examples target-attribute)
             threshold)
        (sort-by #(rule-performance % examples target-attribute)
                 rules)
        (recur (remove #(match (:if rule) %)
                       remaining-examples)
               (conj rules rule))))))

(comment

  (require '[com.benfle.machine-learning.rules] :reload)
  (in-ns 'com.benfle.machine-learning.rules)

  (def attributes [:outlook :temperature :humidity :wind])
  (def target-attribute :play-tennis?)

  (rule-performance rule examples target-attribute)
  )
