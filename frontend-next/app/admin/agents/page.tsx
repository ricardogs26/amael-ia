Para resolver el problema de desbordamiento del texto en el cuadro Registry en vivo de la pestaña de agentes, necesitamos aplicar estilos CSS para truncar el texto de la descripción. Esto se puede hacer usando combinaciones de `overflow`, `text-overflow`, y `white-space` o `maxHeight` con `WebkitLineClamp`.

Vamos a crear la rama `fix/agent-card-description-overflow` desde la rama `develop`, aplicar el fix y abrir un Pull Request.

### Paso 1: Crear la rama
Primero, clona el repositorio y navega al directorio del proyecto:

```sh
git clone https://github.com/ricardogs26/amael-ia.git
cd amael-ia
git checkout develop
git checkout -b fix/agent-card-description-overflow
```

### Paso 2: Aplicar el fix
Abre el archivo `frontend-next/app/admin/agents/page.tsx` y modifica la sección donde se renderiza la descripción del agente. Asegúrate de agregar los estilos CSS necesarios para truncar el texto.

Aquí te dejo un ejemplo de cómo puedes hacerlo:

```tsx
import React from 'react';

// ... (otros imports)

const AgentCard = ({ agent }) => {
  return (
    <div style={{ background: 'var(--bg-surface)', padding: '16px', borderRadius: '8px', border: '1px solid var(--border)', marginBottom: '16px', maxWidth: '400px' }}>
      <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '8px' }}>{agent.name}</div>
      <div style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '8px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{agent.description}</div>
      {/* Otros elementos de la tarjeta */}
    </div>
  );
};

export default function AgentsPage() {
  // ... (otros componentes y lógica)

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <h1 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '20px' }}>Registry en vivo</h1>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
        {agents.map(agent => (
          <AgentCard key={agent.id} agent={agent} />
        ))}
      </div>
    </div>
  );
}
```

### Paso 3: Hacer el commit
Después de modificar el archivo, realiza el commit con el mensaje apropiado:

```sh
git add frontend-next/app/admin/agents/page.tsx
git commit -m "fix: truncate agent description in Registry live card"
```

### Paso 4: Push y abrir Pull Request
Finalmente, realiza el push de los cambios a la rama y abre un Pull Request hacia la rama `develop`:

```sh
git push origin fix/agent-card-description-overflow
```

Ahora, ve a la página del repositorio en GitHub y abre un Pull Request desde la rama `fix/agent-card-description-overflow` hacia la rama `develop`.

### Paso 5: Verificar
Asegúrate de que el texto de la descripción en la tarjeta del Registry en vivo de la pestaña de agentes esté truncado correctamente y no se desborde del contenedor.

¡Eso es todo! Ahora tu Pull Request debería estar listo para ser revisado y fusionado en la rama `develop`.