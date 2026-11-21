using UnityEngine;
using UnityEngine.SceneManagement;

public class RedStickController : MonoBehaviour
{
    public Transform Player;       // Le joueur à poursuivre
    public float MoveSpeed = 3f;   // Vitesse du bâton rouge

    // Niveau 5 : Allongement horizontal aléatoire
    public bool CanStretch = false;      // À cocher uniquement dans Level5 et suivants
    public float MinWidth = 0.5f;        // Largeur minimale sur X
    public float MaxWidth = 4.5f;        // Largeur maximale sur X
    public float StretchSpeed = 3f;      // Vitesse d'allongement / rétrécissement
    public float ChangeInterval = 1.5f;  // Temps entre chaque nouvelle taille aléatoire

    private float _baseMoveSpeed;
    private float _targetWidth;
    private float _stretchTimer = 0f;

    private void Start()
    {
        _baseMoveSpeed = MoveSpeed;
        _targetWidth = transform.localScale.x;
    }

    private void Update()
    {
        // 1. Poursuite du joueur
        if (Player != null)
        {
            Vector3 targetPosition = new Vector3(Player.position.x, transform.position.y, Player.position.z);
            transform.position = Vector3.MoveTowards(transform.position, targetPosition, MoveSpeed * Time.deltaTime);
            transform.LookAt(targetPosition);
        }

        // 2. Allongement horizontal aléatoire (Niveau 5)
        if (CanStretch == true)
        {
            _stretchTimer = _stretchTimer + Time.deltaTime;

            // Toutes les X secondes, on choisit une nouvelle largeur au hasard
            if (_stretchTimer >= ChangeInterval)
            {
                _targetWidth = Random.Range(MinWidth, MaxWidth);
                _stretchTimer = 0f;
            }

            // On fait évoluer la largeur X en douceur vers la taille cible
            float newX = Mathf.MoveTowards(transform.localScale.x, _targetWidth, StretchSpeed * Time.deltaTime);
            transform.localScale = new Vector3(newX, transform.localScale.y, transform.localScale.z);
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