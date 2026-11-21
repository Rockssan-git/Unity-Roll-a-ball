
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class Rollballagent : Agent
{
    public float speed = 10f;

    private Rigidbody rb;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
    }

    // Observations : position et vitesse du joueur
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.position.x);
        sensor.AddObservation(transform.position.z);
        sensor.AddObservation(rb.linearVelocity.x);
        sensor.AddObservation(rb.linearVelocity.z);
    }

    // Actions de déplacement
    public override void OnActionReceived(ActionBuffers actions)
    {
        float horizontal = actions.ContinuousActions[0];
        float vertical = actions.ContinuousActions[1];

        Vector3 movement = new Vector3(horizontal, 0f, vertical);

        rb.AddForce(movement * speed);

        if (Time.frameCount % 30 == 0)
        {
            Debug.Log(
                $"Agent : X={horizontal:F1}, Z={vertical:F1}"
            );
        }
    }

    // Déplacement avec les flèches directionnelles
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        float horizontal = 0f;
        float vertical = 0f;

        if (Input.GetKey(KeyCode.LeftArrow))
            horizontal = -1f;

        if (Input.GetKey(KeyCode.RightArrow))
            horizontal = 1f;

        if (Input.GetKey(KeyCode.UpArrow))
            vertical = 1f;

        if (Input.GetKey(KeyCode.DownArrow))
            vertical = -1f;

        // Correction de l'erreur CS1612
        var continuousActions = actionsOut.ContinuousActions;

        continuousActions[0] = horizontal;
        continuousActions[1] = vertical;
    }
}
