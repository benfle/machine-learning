(ns com.benfle.machine-learning.decision-tree
  "From Chapter 3 of Tom Mitchell's \"Machine Learning\".")

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

;; id3

(defn entropy
  "The entropy of the set of examples relative to this target attribute."
  [examples target-attribute]
  (->> examples
       (map target-attribute)
       frequencies
       vals
       (map #(double (/ % (count examples))))
       (map #(* % (/ (Math/log %)
                     (Math/log 2))))
       (reduce - 0)))

(defn gain
  "The information gain of this attribute."
  [examples target-attribute attribute]
  (- (entropy examples target-attribute)
     (reduce + 0 (map (fn [value]
                        (let [sv (filter #(= value (get % attribute))
                                         examples)]
                          (* (/ (count sv)
                                (count examples))
                             (entropy sv target-attribute))))
                      (set (map attribute examples))))))

(defn most-common-value
  [examples attribute]
  (->> examples
       (map attribute)
       frequencies
       (sort-by val)
       last
       first))

(defn id3
  [examples target-attribute attributes]
  (cond
    (every? true? (map target-attribute examples))
    {:label true}

    (every? false? (map target-attribute examples))
    {:label false}

    (empty? attributes)
    {:label (most-common-value examples target-attribute)}

    :else
    (let [gains (->> (for [attribute attributes]
                       [attribute (gain examples target-attribute attribute)])
                     (sort-by second))
          attribute (first (last gains))]
      {:attribute attribute
       :values (into {} (for [value (set (map attribute examples))]
                          (let [sub-examples (filter #(= value (attribute %))
                                                     examples)]
                            [value
                             (if (empty? sub-examples)
                               {:label (most-common-value examples target-attribute)}
                               (id3 sub-examples target-attribute (disj attributes attribute)))])))})))

(comment

  (require '[com.benfle.machine-learning.decision-tree] :reload)
  (in-ns 'com.benfle.machine-learning.decision-tree)

  (clojure.pprint/pprint
   (id3 examples :play-tennis? #{:outlook :temperature :humidity :wind}))

  )
