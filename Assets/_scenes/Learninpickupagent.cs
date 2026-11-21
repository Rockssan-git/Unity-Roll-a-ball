
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class LearningPickupAgent : Agent
{
    [Header("Environnement")]
    public Transform targetPickup;
    public float speed = 10f;
    public int maxStepsPerEpisode = 250;

    [Header("Apprentissage Q-Learning")]
    public bool trainingMode = true;
    public float learningRate = 0.2f;
    public float gamma = 0.95f;
    public float epsilon = 1f;
    public float epsilonDecay = 0.997f;
    public float minEpsilon = 0.05f;

    // 3 valeurs pour chacune des 4 observations
    // 3^4 = 81 etats, 5 actions possibles
    private const int StateCount = 81;
    private const int ActionCount = 5;

    private float[,] qTable =
        new float[StateCount, ActionCount];

    private Rigidbody rb;
    private Vector3 startPosition;
    private Quaternion startRotation;
    private Vector3 startPickupPosition;

    private int previousState;
    private int previousAction;
    private float previousDistance;
    private bool hasPreviousAction;

    private bool episodeFinished;
    private int episodeSteps;
    private int episodes;
    private int successes;
    private float episodeReward;

    // Historique des 20 derniers episodes
    private int[] recentResults = new int[20];
    private int recentCount = 0;
    private int recentIndex = 0;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();

        if (targetPickup == null || rb == null)
        {
            Debug.LogError(
                "QLearning : Rigidbody ou Target Pickup manquant !"
            );
            enabled = false;
            return;
        }

        startPosition = transform.position;
        startRotation = transform.rotation;
        startPickupPosition = targetPickup.position;

        // Facilite le freinage de la boule
        rb.linearDamping = 1.5f;

        Debug.Log("Q-Learning initialise !");
    }

    public override void OnEpisodeBegin()
    {
        if (rb == null || targetPickup == null)
            return;

        episodeFinished = false;
        episodeSteps = 0;
        episodeReward = 0f;
        hasPreviousAction = false;

        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        transform.SetPositionAndRotation(
            startPosition,
            startRotation
        );

        targetPickup.position = startPickupPosition;
        targetPickup.gameObject.SetActive(true);
    }

    // Conserver les 4 observations ML-Agents
    public override void CollectObservations(
        VectorSensor sensor)
    {
        Vector3 direction =
            targetPickup.position - transform.position;

        sensor.AddObservation(direction.x);
        sensor.AddObservation(direction.z);
        sensor.AddObservation(rb.linearVelocity.x);
        sensor.AddObservation(rb.linearVelocity.z);
    }

    // Transformer une valeur continue en 3 categories
    private int Discretize(float value, float threshold)
    {
        if (value < -threshold) return 0;
        if (value > threshold) return 2;
        return 1;
    }

    // Construire un etat parmi les 81
    private int GetState()
    {
        Vector3 direction =
            targetPickup.position - transform.position;

        int dx = Discretize(direction.x, 0.7f);
        int dz = Discretize(direction.z, 0.7f);

        int vx = Discretize(rb.linearVelocity.x, 0.5f);
        int vz = Discretize(rb.linearVelocity.z, 0.5f);

        return (((dx * 3 + dz) * 3 + vx) * 3 + vz);
    }

    private float GetDistance()
    {
        Vector3 direction =
            targetPickup.position - transform.position;

        return new Vector2(
            direction.x, direction.z
        ).magnitude;
    }

    // Meilleure valeur Q de l'etat suivant
    private float GetMaxQ(int state)
    {
        float best = float.NegativeInfinity;

        for (int a = 0; a < ActionCount; a++)
        {
            best = Mathf.Max(best, qTable[state, a]);
        }

        return best;
    }

    // Choisir entre exploration et exploitation
    private int ChooseAction(int state)
    {
        if (trainingMode && Random.value < epsilon)
        {
            return Random.Range(0, ActionCount);
        }

        float bestValue = float.NegativeInfinity;
        int bestAction = 0;
        int ties = 0;

        for (int a = 0; a < ActionCount; a++)
        {
            float value = qTable[state, a];

            if (value > bestValue + 0.000001f)
            {
                bestValue = value;
                bestAction = a;
                ties = 1;
            }
            else if (Mathf.Abs(value - bestValue) <= 0.000001f)
            {
                ties++;

                if (Random.Range(0, ties) == 0)
                    bestAction = a;
            }
        }

        return bestAction;
    }

    // Equation du Q-Learning
    private void UpdateQ(
        int state,
        int action,
        float reward,
        int nextState,
        bool terminal)
    {
        if (!trainingMode)
            return;

        float oldQ = qTable[state, action];

        float targetQ = reward;

        if (!terminal)
        {
            targetQ += gamma * GetMaxQ(nextState);
        }

        qTable[state, action] =
            oldQ + learningRate * (targetQ - oldQ);
    }

    // Apprendre du resultat de l'action precedente
    private float LearnPreviousAction(bool terminal, float bonus)
    {
        float reward = bonus;

        if (hasPreviousAction)
        {
            float progress =
                previousDistance - GetDistance();

            reward += Mathf.Clamp(
                progress * 0.5f,
                -0.1f,
                0.1f
            );

            reward -= 0.002f;

            UpdateQ(
                previousState,
                previousAction,
                reward,
                GetState(),
                terminal
            );
        }

        AddReward(reward);
        episodeReward += reward;

        return reward;
    }

    // Les actions sont maintenant choisies par Q-Learning
    public override void Heuristic(
        in ActionBuffers actionsOut)
    {
        var actions = actionsOut.ContinuousActions;

        actions[0] = 0f;
        actions[1] = 0f;

        if (episodeFinished || rb == null ||
            targetPickup == null)
            return;

        if (hasPreviousAction)
        {
            LearnPreviousAction(false, 0f);
        }

        int state = GetState();
        int action = ChooseAction(state);

        previousState = state;
        previousAction = action;
        previousDistance = GetDistance();
        hasPreviousAction = true;

        // 0 = aucune force
        // 1 = gauche
        // 2 = droite
        // 3 = avant
        // 4 = arriere

        switch (action)
        {
            case 1:
                actions[0] = -1f;
                break;

            case 2:
                actions[0] = 1f;
                break;

            case 3:
                actions[1] = 1f;
                break;

            case 4:
                actions[1] = -1f;
                break;
        }
    }

    // Executer physiquement les actions
    public override void OnActionReceived(
        ActionBuffers actions)
    {
        if (episodeFinished || rb == null)
            return;

        float x = actions.ContinuousActions[0];
        float z = actions.ContinuousActions[1];

        rb.AddForce(
            new Vector3(x, 0f, z) * speed
        );

        episodeSteps++;

        // Chute
        if (transform.position.y < -1f)
        {
            FinishEpisode(false, -1f);
        }

        // Temps limite
        else if (episodeSteps >= maxStepsPerEpisode)
        {
            FinishEpisode(false, -0.5f);
        }
    }

    // Detecter la collecte du cube
    private void OnTriggerEnter(Collider other)
    {
        CheckPickup(other.transform);
    }

    private void OnCollisionEnter(Collision collision)
    {
        CheckPickup(collision.transform);
    }

    private void CheckPickup(Transform touched)
    {
        if (episodeFinished || targetPickup == null)
            return;

        if (touched == targetPickup ||
            touched.IsChildOf(targetPickup))
        {
            FinishEpisode(true, 2f);
        }
    }

    private void FinishEpisode(
        bool success,
        float terminalReward)
    {
        if (episodeFinished)
            return;

        episodeFinished = true;

        LearnPreviousAction(true, terminalReward);

        episodes++;

        if (success)
            successes++;

        // Conserver les 20 derniers resultats
        recentResults[recentIndex] = success ? 1 : 0;
        recentIndex = (recentIndex + 1) % 20;
        recentCount = Mathf.Min(recentCount + 1, 20);

        int recentSuccesses = 0;

        for (int i = 0; i < recentCount; i++)
            recentSuccesses += recentResults[i];

        float recentRate = recentCount > 0
            ? 100f * recentSuccesses / recentCount
            : 0f;

        if (trainingMode)
        {
            epsilon = Mathf.Max(
                minEpsilon,
                epsilon * epsilonDecay
            );
        }

        // Afficher les resultats
        if (episodes % 10 == 0 || success)
        {
            Debug.Log(
                $"Q-Learning | Episode {episodes} | " +
                $"Succes={success} | " +
                $"Reward={episodeReward:F2} | " +
                $"Epsilon={epsilon:F3} | " +
                $"Total={successes}/{episodes} | " +
                $"Reussite recente={recentRate:F1}% | " +
                $"Mode={(trainingMode ? "TRAIN" : "TEST")}"
            );
        }

        EndEpisode();
    }
}
