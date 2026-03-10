#!/bin/bash
# Script para limpiar imágenes antiguas, conservando en uso y las 5 más recientes

IMAGE_NAME="registry.richardx.dev/whatsapp-bridge"
KEEP_COUNT=5

echo "======================================================"
echo "🧹 Limpieza de imágenes Docker locales"
echo "📦 Imagen: $IMAGE_NAME"
echo "======================================================"

# 1. Obtener imágenes en uso por Kubernetes en todos los namespaces
echo -e "\n🔍 Detectando imágenes de $IMAGE_NAME en uso en el cluster Kubernetes..."
IMAGES_IN_USE=$(kubectl get pods -A -o jsonpath='{range .items[*].spec.containers[*]}{.image}{"\n"}{end}' | sort | uniq | grep "$IMAGE_NAME")

if [ -z "$IMAGES_IN_USE" ]; then
    echo "⚠️  (No se detectaron pods en kubernetes usando versiones de esta imagen actualmente)"
else
    echo "💡 Imágenes actualmente en uso en el clúster:"
    echo "$IMAGES_IN_USE" | sed 's/^/   - /'
fi

# 2. Obtener TODAS las imágenes locales de ese repositorio
# Docker por defecto las ordena de más nuevas a más antiguas
AVAILABLE_IMAGES=$(docker image ls --format '{{.Repository}}:{{.Tag}}' "$IMAGE_NAME" | grep -v "<none>")

if [ -z "$AVAILABLE_IMAGES" ]; then
   echo -e "\nNo se encontraron imágenes locales de $IMAGE_NAME."
   exit 0
fi

TOTAL_FOUND=$(echo "$AVAILABLE_IMAGES" | wc -l)
echo -e "\n📊 Total de versiones locales encontradas: $TOTAL_FOUND"

# 3. Extraer las N más recientes que queremos mantener
RECENT_IMAGES=$(echo "$AVAILABLE_IMAGES" | head -n "$KEEP_COUNT")

# 4. Recorrer todas y decidir si se eliminan o se mantienen
echo -e "\n⚙️  Evaluando imágenes para eliminación..."
for IMG in $AVAILABLE_IMAGES; do
    IN_USE=false
    RECENT=false
    
    # Comprobar si está en uso en K8s
    if echo "$IMAGES_IN_USE" | grep -q "^${IMG}$"; then
        IN_USE=true
    fi
    
    # Comprobar si es de las recientes
    if echo "$RECENT_IMAGES" | grep -q "^${IMG}$"; then
        RECENT=true
    fi
    
    # Decisión
    if [ "$IN_USE" = true ]; then
        echo "✅ Mantenida: $IMG (Está en uso en Kubernetes)"
    elif [ "$RECENT" = true ]; then
        echo "✅ Mantenida: $IMG (Es una de las $KEEP_COUNT más recientes)"
    else
        echo "🗑️  Eliminando: $IMG"
        # Forzamos con un silenciador de errores por si está vinculada a un contenedor parado
        docker rmi "$IMG" 2>/dev/null || echo "   ⚠️ No se pudo eliminar (puede estar usada por un contenedor local que fue detenido)"
    fi
done

# 5. Opcional: Limpiar las imágenes "dangling" (fragmentos temporales de build de esta imagen)
echo -e "\n♻️  Limpiando fragmentos de compilación (dangling)..."
DANGLING=$(docker images "$IMAGE_NAME" -f "dangling=true" -q)
if [ -n "$DANGLING" ]; then
    docker rmi $DANGLING 2>/dev/null
    echo "✔️  Fragmentos eliminados."
fi

echo -e "\n✅ ¡Limpieza de imágenes completada!\n"
