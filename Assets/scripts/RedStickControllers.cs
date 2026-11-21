using UnityEngine;
using UnityEngine.SceneManagement;

public class RedStickController : MonoBehaviour
{
    public Transform Player;       // Le joueur à poursuivre
    public float MoveSpeed = 3f;   // Vitesse du bâton rouge
    //added a private variable to store the base move speed START
    private float _baseMoveSpeed;

    private void Start()
    {
        _baseMoveSpeed = MoveSpeed;
    }
    //added a private variable to store the base move speed END
    private void Update()
    {
        if (Player != null)
        {
            // Position cible (on garde la hauteur Y du bâton)
            Vector3 targetPosition = new Vector3(Player.position.x, transform.position.y, Player.position.z);

            // Déplace le bâton vers le joueur
            transform.position = Vector3.MoveTowards(transform.position, targetPosition, MoveSpeed * Time.deltaTime);

            // Oriente le bâton vers le joueur
            transform.LookAt(targetPosition);
        }
    }

    // Si le bâton touche le joueur, on recommence le niveau
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Player"))
        {
            SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex);
        }
    }
    // Niveau 3 : modifier temporairement la vitesse du bâton rouge
    public void ChangeSpeed(float multiplier, float duration)
    {
        MoveSpeed = _baseMoveSpeed * multiplier;
        CancelInvoke("ResetSpeed");
        Invoke("ResetSpeed", duration);
    }

    private void ResetSpeed()
    {
        MoveSpeed = _baseMoveSpeed;
    }
}