
using UnityEngine;

public class MovingPickup : MonoBehaviour
{
    // Direction du mouvement
    public enum MovementAxis
    {
        X,
        Z
    }

    [Header("Configuration du mouvement")]
    public bool enableMovement = true;

    public MovementAxis movementAxis = MovementAxis.X;

    [Min(0f)]
    public float moveDistance = 1f;

    [Min(0f)]
    public float moveSpeed = 0.25f;

    // Position de depart du pickup
    private Vector3 initialPosition;

    // Temps ecoule
    private float elapsedTime = 0f;

    private void Awake()
    {
        initialPosition = transform.localPosition;
    }

    private void OnEnable()
    {
        // Reinitialisation lors du retour du cube
        elapsedTime = 0f;
        transform.localPosition = initialPosition;
    }

    private void FixedUpdate()
    {
        if (!enableMovement)
            return;

        // Faire avancer le temps
        elapsedTime += Time.fixedDeltaTime;

        // Calcul du mouvement oscillatoire
        float movement = Mathf.Sin(
            elapsedTime * moveSpeed * 2f * Mathf.PI
        ) * moveDistance;

        Vector3 newPosition = initialPosition;

        // Choisir l'axe du deplacement
        if (movementAxis == MovementAxis.X)
        {
            newPosition.x += movement;
        }
        else
        {
            newPosition.z += movement;
        }

        // Deplacer le cube
        transform.localPosition = newPosition;
    }
}
