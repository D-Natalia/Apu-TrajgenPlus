import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, View, Text, Button, Alert, Modal, FlatList, TouchableOpacity } from 'react-native';
import MapView, { Polyline, Marker } from 'react-native-maps';
import * as Location from 'expo-location';
import AsyncStorage from '@react-native-async-storage/async-storage';

export interface Coordinate {
  lat: number;
  lon: number;
  timestamp?: number;
}

/**
 * Wrapper around fetch to include a timeout mechanism.
 * @param resource The URL to fetch.
 * @param options Fetch options, including an optional `timeout` in milliseconds.
 * @returns The fetch Response promise.
 */
async function fetchWithTimeout(resource: RequestInfo, options: RequestInit & { timeout?: number } = {}) {
  const { timeout = 15000 } = options; // Default timeout 8 seconds

  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  const response = await fetch(resource, {
    ...options,
    signal: controller.signal  
  });
  clearTimeout(id);
  return response;
}

export interface ObfuscateResponse {
  user_id: string;
  obfuscated_route: Coordinate[];
  generated_point?: Coordinate;
  generated_route?: Coordinate[];
  k_steps: number;
}

export interface HistoryItem {
  id: string;
  date: string;
  realRoute: Coordinate[];
  syntheticRoute: Coordinate[]; 
  generatedRoute: Coordinate[]; 
  kSteps: number | null;
  suMethod?: string;
  protectionInterval?: string;
}

export default function HomeScreen() {
  const [isTracking, setIsTracking] = useState(false);
  const [realRoute, setRealRoute] = useState<Coordinate[]>([]);
  const [syntheticRoute, setSyntheticRoute] = useState<Coordinate[]>([]);
  const [generatedRoute, setGeneratedRoute] = useState<Coordinate[]>([]);
  const [kSteps, setKSteps] = useState<number | null>(null);
  const [currentLoc, setCurrentLoc] = useState<Coordinate | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');
  const [routeStyle, setRouteStyle] = useState<'lines' | 'points'>('lines');
  const [routeVariant, setRouteVariant] = useState<'osrm' | 'generated' | 'both'>('osrm');
  const [suMethod, setSuMethod] = useState<string>('1');
  const [protectionInterval, setProtectionInterval] = useState<string>('80-155');
  const [showSuPicker, setShowSuPicker] = useState(false);
  const [showIntervalPicker, setShowIntervalPicker] = useState(false);
  const [showControls, setShowControls] = useState(true);
  const [editingHistoryItem, setEditingHistoryItem] = useState<HistoryItem | null>(null);

  const locationSubscription = useRef<Location.LocationSubscription | null>(null);

  useEffect(() => {
    (async () => {
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permisiune refuzată', 'Aplicația are nevoie de acces la locație.');
        return;
      }
      let location = await Location.getCurrentPositionAsync({});
      setCurrentLoc({ lat: location.coords.latitude, lon: location.coords.longitude });
    })();

    (async () => {
      try {
        const savedHistory = await AsyncStorage.getItem('routesHistory');
        if (savedHistory) {
          setHistory(JSON.parse(savedHistory));
        }
      } catch (e) {
        console.error('Eroare la încărcarea istoricului', e);
      }
    })();
  }, []);

  const startTracking = async () => {
    setRealRoute([]);
    setSyntheticRoute([]);
    setGeneratedRoute([]);
    setKSteps(null);
    setEditingHistoryItem(null); 
    setIsTracking(true);

    const newSessionId = Date.now().toString();
    setSessionId(newSessionId);

    locationSubscription.current = await Location.watchPositionAsync(
      {
        accuracy: Location.Accuracy.BestForNavigation,
        timeInterval: 5000, 
        distanceInterval: 2,
      },
      async (loc) => {
        const newCoord = { lat: loc.coords.latitude, lon: loc.coords.longitude, timestamp: loc.timestamp / 1000 };
        setRealRoute((prev) => [...prev, newCoord]);
        setCurrentLoc(newCoord);

        try {
          const response = await fetchWithTimeout('http://10.191.95.85:8000/obfuscate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              user_id: newSessionId,
              lat: loc.coords.latitude,
              lng: loc.coords.longitude,
              timestamp: loc.timestamp / 1000,
              su_method: suMethod,
              protection_interval: protectionInterval,
            }),
            timeout: 120000, // 12 secunde timeout
          });

          const data: ObfuscateResponse = await response.json();
          if (data.obfuscated_route && data.obfuscated_route.length > 0) {
            setSyntheticRoute((prev) => [...prev, ...data.obfuscated_route]);
            setKSteps(data.k_steps);
          }
          if (data.generated_point) {
            setGeneratedRoute((prev) => [...prev, data.generated_point!]);
          }
        } catch (error) {
          if (error.name === 'AbortError') {
            console.warn('Cererea API a expirat (timeout).');
          } else {
            console.error('Eroare de conexiune API:', error);
          }
        }
      }
    );
  };

  const stopTracking = async () => {
    if (locationSubscription.current) {
      locationSubscription.current.remove();
      locationSubscription.current = null;
    }
    setIsTracking(false);

    if (realRoute.length < 2) {
      Alert.alert('Date insuficiente', 'Ruta este prea scurtă pentru a fi salvată.');
      return;
    }

    setHistory((prevHistory) => {
      const newItem: HistoryItem = {
        id: sessionId || Date.now().toString(),
        date: new Date().toLocaleString(),
        realRoute,
        syntheticRoute,
        generatedRoute,
        kSteps,
        suMethod,
        protectionInterval,
      };
      const newHistory = [newItem, ...prevHistory];
      AsyncStorage.setItem('routesHistory', JSON.stringify(newHistory)).catch((e) => console.error(e));
      return newHistory;
    });
  };

  const loadFromHistory = (item: HistoryItem) => {
    setRealRoute(item.realRoute);
    setSyntheticRoute(item.syntheticRoute);
    setGeneratedRoute(item.generatedRoute || []);
    setKSteps(item.kSteps);
    setSuMethod(item.suMethod || '1');
    setProtectionInterval(item.protectionInterval || '80-155');
    setShowHistory(false);
    setShowControls(true); 
    setEditingHistoryItem(item); 
    if (item.realRoute.length > 0) {
      setCurrentLoc(item.realRoute[0]);
    }
  };

  const regenerateRoute = async (item: HistoryItem, useCurrentSettings: boolean = false) => {
    try {

      console.log(`Încep regenerarea pentru ${item.id} cu SU: ${suMethod} și Interval: ${protectionInterval}`);

      const response = await fetchWithTimeout('http://10.191.95.85:8000/obfuscate_route', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: `regen_${item.id}_${Date.now()}`, 
          route: item.realRoute,
          su_method: useCurrentSettings ? suMethod : item.suMethod || '1',
          protection_interval: useCurrentSettings ? protectionInterval : item.protectionInterval || '80-155',
        }),
        timeout: 120000, 
      });
      console.log("Am primit răspuns:", response.status);

      const data = await response.json();
      console.log("JSON citit");
      if (data.obfuscated_route && data.generated_route) {
        const newSyntheticRoute = data.obfuscated_route;
        const newGeneratedRoute = data.generated_route;

        const updatedItem: HistoryItem = {
          ...item,
          syntheticRoute: newSyntheticRoute,
          generatedRoute: newGeneratedRoute,
          kSteps: data.k_steps,
          suMethod: useCurrentSettings ? suMethod : item.suMethod || '1',
          protectionInterval: useCurrentSettings ? protectionInterval : item.protectionInterval || '80-155',
        };

        setHistory((prev) => {
          const updated = prev.map((h) =>
            h.id === item.id ? updatedItem : h
          );
          AsyncStorage.setItem('routesHistory', JSON.stringify(updated)).catch((e) => console.error(e));
          return updated;
        });
        // Actualizăm și vizualizarea curentă
        setSyntheticRoute(newSyntheticRoute);
        setGeneratedRoute(newGeneratedRoute);
        setKSteps(data.k_steps);
        setEditingHistoryItem(updatedItem); // Actualizăm elementul în curs de editare
        Alert.alert('Succes', 'Ruta sintetică a fost regenerată cu succes!');
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        Alert.alert('Eroare', 'Cererea de regenerare a expirat (timeout). Serverul ar putea fi prea lent.');
      } else {
        console.error(error);
        Alert.alert('Eroare', 'Conexiunea la server a eșuat.');
      }
    }
  };

  const deleteHistoryItem = async (id: string) => {
    Alert.alert('Confirmare', 'Ești sigur că vrei să ștergi această rută din istoric?', [
      { text: 'Anulează', style: 'cancel' },
      {
        text: 'Șterge',
        style: 'destructive',
        onPress: async () => {
          const updatedHistory = history.filter((item) => item.id !== id);
          setHistory(updatedHistory);
          await AsyncStorage.setItem('routesHistory', JSON.stringify(updatedHistory));
        },
      },
    ]);
  };

  const clearHistory = async () => {
    Alert.alert('Confirmare', 'Ești sigur că vrei să ștergi TOT istoricul?', [
      { text: 'Anulează', style: 'cancel' },
      {
        text: 'Șterge Tot',
        style: 'destructive',
        onPress: async () => {
          setHistory([]);
          await AsyncStorage.removeItem('routesHistory');
        },
      },
    ]);
  };

  const variantLabel = routeVariant === 'osrm' ? 'OSRM' : routeVariant === 'generated' ? 'Generate' : 'Ambele';

  return (
    <View style={styles.container}>
      <MapView
        style={styles.map}
        region={
          currentLoc
            ? {
                latitude: currentLoc.lat,
                longitude: currentLoc.lon,
                latitudeDelta: 0.005,
                longitudeDelta: 0.005,
              }
            : undefined
        }
        showsUserLocation={true}
      >
        {routeStyle === 'lines' && realRoute.length > 0 && (
          <Polyline
            coordinates={realRoute.map((c) => ({ latitude: c.lat, longitude: c.lon }))}
            strokeColor="#0000FF"
            strokeWidth={5}
          />
        )}
        {routeStyle === 'points' &&
          realRoute.map((c, idx) => (
            <Marker key={`real-${idx}`} coordinate={{ latitude: c.lat, longitude: c.lon }} anchor={{ x: 0.5, y: 0.5 }}>
              <View style={{ width: 8, height: 8, borderRadius: 4, backgroundColor: '#0000FF' }} />
            </Marker>
          ))}

        {(routeVariant === 'osrm' || routeVariant === 'both') && routeStyle === 'lines' && syntheticRoute.length > 0 && (
          <Polyline
            coordinates={syntheticRoute.map((c) => ({ latitude: c.lat, longitude: c.lon }))}
            strokeColor="#FF0000"
            strokeWidth={5}
          />
        )}
        {(routeVariant === 'osrm' || routeVariant === 'both') && routeStyle === 'points' &&
          syntheticRoute.map((c, idx) => (
            <Marker key={`synth-${idx}`} coordinate={{ latitude: c.lat, longitude: c.lon }} anchor={{ x: 0.5, y: 0.5 }}>
              <View style={{ width: 8, height: 8, borderRadius: 4, backgroundColor: '#FF0000' }} />
            </Marker>
          ))}

        {(routeVariant === 'generated' || routeVariant === 'both') && routeStyle === 'lines' && generatedRoute.length > 0 && (
          <Polyline
            coordinates={generatedRoute.map((c) => ({ latitude: c.lat, longitude: c.lon }))}
            strokeColor="#FFA500"
            strokeWidth={4}
          />
        )}
        {(routeVariant === 'generated' || routeVariant === 'both') && routeStyle === 'points' &&
          generatedRoute.map((c, idx) => (
            <Marker key={`gen-${idx}`} coordinate={{ latitude: c.lat, longitude: c.lon }} anchor={{ x: 0.5, y: 0.5 }}>
              <View style={{ width: 8, height: 8, borderRadius: 4, backgroundColor: '#FFA500' }} />
            </Marker>
          ))}
      </MapView>

      {showControls && (
      <View style={styles.controls}>
        <Text style={styles.infoText}>
          Puncte Reale: {realRoute.length} | OSRM: {syntheticRoute.length} | Generate: {generatedRoute.length}
        </Text>
        {kSteps !== null && <Text style={styles.suText}>k-steps adaptiv: {kSteps}</Text>}

        <Button title={isTracking ? 'Oprește Tracking' : 'Pornește Tracking'} color={isTracking ? 'red' : 'green'} onPress={isTracking ? stopTracking : startTracking} />
        <View style={{ marginTop: 10, width: '100%' }}>
          <Button
            title={`Aspect Vizual: ${routeStyle === 'lines' ? 'Linii' : 'Puncte'}`}
            onPress={() => setRouteStyle((prev) => (prev === 'lines' ? 'points' : 'lines'))}
            color="#007BFF"
          />
        </View>

        <View style={{ marginTop: 10, width: '100%' }}>
          <Button
            title={`Variantă: ${variantLabel}`}
            onPress={() => setRouteVariant((prev) => (prev === 'osrm' ? 'generated' : prev === 'generated' ? 'both' : 'osrm'))}
            color="#009688"
          />
        </View>

        <View style={{ marginTop: 10, width: '100%' }}>
          <Button title={`Metodă calcul SU: ${suMethod}`} onPress={() => setShowSuPicker(true)} color="#8E44AD" />
        </View>

        <View style={{ marginTop: 10, width: '100%' }}>
          <Button title={`Interval Protecție: ${protectionInterval}m`} onPress={() => setShowIntervalPicker(true)} color="#f39c12" />
        </View>

        <View style={{ marginTop: 10, width: '100%' }}>
          <Button title={`Istoric (${history.length})`} onPress={() => setShowHistory(true)} color="#555" />
        </View>

        <View style={{ marginTop: 15, width: '100%', alignItems: 'center' }}>
          <Button title="Închide meniu" color="#444" onPress={() => setShowControls(false)} />
        </View>
      </View>
      )}

      {/* Compact overlay when controls are hidden */}
      {!showControls && (
        <View style={styles.compactOverlay}>
          <View style={styles.compactRow}>
            <Text style={styles.compactText}>k: {kSteps !== null ? kSteps : '-'}</Text>
            <Text style={styles.compactText}>Real: {realRoute.length}</Text>
            <Text style={styles.compactText}>OSRM: {syntheticRoute.length}</Text>
            <Text style={styles.compactText}>Gen: {generatedRoute.length}</Text>
          </View>
          <View style={styles.compactRow}>
            <Button title={isTracking ? 'Oprește' : 'Pornește'} color={isTracking ? 'red' : 'green'} onPress={isTracking ? stopTracking : startTracking} />
            <View style={{ width: 8 }} />
            <Button title='Meniu' color='#007BFF' onPress={() => setShowControls(true)} />
          </View>
        </View>
      )}

      <Modal visible={showHistory} animationType="slide" onRequestClose={() => setShowHistory(false)}>
        <View style={styles.modalContainer}>
          <Text style={styles.modalTitle}>Istoric Rute</Text>
          <FlatList
            data={history}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <View style={styles.historyItem}>
                <View style={{ flex: 1, paddingRight: 10 }}>
                  <Text style={styles.historyDate}>{item.date}</Text>
                  <Text>Puncte GPS: {item.realRoute.length}</Text>
                  <Text style={{ color: '#8E44AD', marginTop: 2, fontWeight: 'bold' }}>Metoda SU: {item.suMethod || '1'}</Text>
                  <Text style={{ color: '#f39c12', marginTop: 2, fontWeight: 'bold' }}>Interval: {item.protectionInterval || '80-155'}m</Text>
                </View>
                <View>
                  <Button title="Deschide" onPress={() => loadFromHistory(item)} />
                  <View style={{ marginTop: 5 }}>
                    <Button title="Regenerează" color="orange" onPress={() => regenerateRoute(item, true)} />
                  </View>
                  <View style={{ marginTop: 5 }}>
                    <Button title="Șterge" color="red" onPress={() => deleteHistoryItem(item.id)} />
                  </View>
                </View>
              </View>
            )}
            ListEmptyComponent={<Text style={{ textAlign: 'center', marginTop: 20 }}>Nicio rută salvată încă.</Text>}
          />
          <View style={styles.modalButtons}>
            <Button title="Șterge tot" color="red" onPress={clearHistory} />
            <Button title="Închide" onPress={() => setShowHistory(false)} />
          </View>
        </View>
      </Modal>

      <Modal visible={showSuPicker} transparent={true} animationType="fade" onRequestClose={() => setShowSuPicker(false)}>
        <View style={styles.pickerModalContainer}>
          <View style={styles.pickerModalContent}>
            <Text style={styles.pickerTitle}>Alege metoda de calcul SU</Text>
            {['1', '2', '3'].map((method) => (
              <TouchableOpacity key={method} style={styles.pickerOption} onPress={() => { setSuMethod(method); setShowSuPicker(false); }}>
                <Text style={styles.pickerOptionText}>Metoda {method}</Text>
              </TouchableOpacity>
            ))}
            <Button title="Anulează" color="red" onPress={() => setShowSuPicker(false)} />
          </View>
        </View>
      </Modal>

      <Modal visible={showIntervalPicker} transparent={true} animationType="fade" onRequestClose={() => setShowIntervalPicker(false)}>
        <View style={styles.pickerModalContainer}>
          <View style={styles.pickerModalContent}>
            <Text style={styles.pickerTitle}>Alege intervalul de protecție</Text>
            {['80-155', '170-350', '250-450', '350-550'].map((interval) => (
              <TouchableOpacity key={interval} style={styles.pickerOption} onPress={() => { setProtectionInterval(interval); setShowIntervalPicker(false); }}>
                <Text style={styles.pickerOptionText}>{interval} m</Text>
              </TouchableOpacity>
            ))}
            <Button title="Anulează" color="red" onPress={() => setShowIntervalPicker(false)} />
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  map: { flex: 1 },
  controls: {
    position: 'absolute',
    bottom: 30,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    padding: 20,
    borderRadius: 15,
    alignItems: 'center',
    elevation: 5,
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 4,
  },
  infoText: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 5,
    color: '#333',
  },
  suText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#d32f2f',
    marginBottom: 15,
  },
  modalContainer: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  modalTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
    marginTop: 40,
  },
  historyItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
    elevation: 2,
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowOffset: { width: 0, height: 1 },
    shadowRadius: 2,
  },
  historyDate: {
    fontWeight: 'bold',
    fontSize: 16,
    marginBottom: 4,
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 20,
    marginBottom: 20,
  },
  pickerModalContainer: {
    flex: 1,
    justifyContent: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 20,
  },
  pickerModalContent: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 20,
    elevation: 5,
  },
  pickerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
    textAlign: 'center',
  },
  pickerOption: {
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
  },
  pickerOptionText: {
    fontSize: 16,
    textAlign: 'center',
    color: '#007BFF',
  },
  compactOverlay: {
    position: 'absolute',
    top: 40,
    right: 20,
    backgroundColor: 'rgba(255,255,255,0.95)',
    padding: 10,
    borderRadius: 10,
    elevation: 6,
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 3,
  },
  compactRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  compactText: {
    fontSize: 12,
    marginHorizontal: 6,
  },
});
