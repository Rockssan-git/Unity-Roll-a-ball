using UnityEngine;

public class SpeedBall : MonoBehaviour
{
    public float SpeedMultiplier = 1.5f;  // 1.5 pour accélérer (+), 0.5 pour ralentir (-)
    public float Duration = 3f;           // Durée de l'effet en secondes

    private void OnTriggerEnter(Collider other)
    {
        // Si c'est le joueur qui touche la boule
        PlayerController player = other.GetComponent<PlayerController>();
        if (player != null)
        {
            player.ChangeSpeed(SpeedMultiplier, Duration);
            Destroy(gameObject);
            return;
        }

        // Si c'est le bâton rouge qui touche la boule
        RedStickController stick = other.GetComponent<RedStickController>();
        if (stick != null)
        {
            stick.ChangeSpeed(SpeedMultiplier, Duration);
            Destroy(gameObject);
        }
    }
}