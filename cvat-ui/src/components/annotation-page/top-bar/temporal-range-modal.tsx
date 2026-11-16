import React, { useState, useEffect } from 'react';
import Modal from 'antd/lib/modal';
import Button from 'antd/lib/button';
import Input from 'antd/lib/input';
import InputNumber from 'antd/lib/input-number';
import Slider from 'antd/lib/slider';
import Radio from 'antd/lib/radio';
import Select from 'antd/lib/select';
import Tag from 'antd/lib/tag';
import Card from 'antd/lib/card';
import Tabs from 'antd/lib/tabs';
import Checkbox from 'antd/lib/checkbox';
import Alert from 'antd/lib/alert';
import Divider from 'antd/lib/divider';
import notification from 'antd/lib/notification';
import Tooltip from 'antd/lib/tooltip';
import Text from 'antd/lib/typography/Text';
import Table from 'antd/lib/table';
import Spin from 'antd/lib/spin';
import {
    ThunderboltFilled,
    RobotFilled,
    RocketFilled,
    SettingFilled,
    PlusOutlined,
    CheckCircleFilled,
    FireFilled,
    TagsFilled,
    ScissorOutlined,
    SwapOutlined,
    BulbFilled,
    DeleteOutlined,
    BarChartOutlined,
    CopyOutlined,
    ReloadOutlined,
    ExperimentOutlined,
    WarningOutlined,
} from '@ant-design/icons';
import { Job } from 'cvat-core-wrapper';

interface Props {
    jobInstance: Job;
    frameNumber: number;
}

interface ModelItem {
    name: string;
    filename: string;
    size_mb: number;
    modified: string;
    is_active: boolean;
}

interface LabelInfo {
    id: number;
    name: string;
    color?: string;
}

interface ShapeItem {
    id: number;
    frame: number;
    label_id: number;
    label_name: string;
    points: number[];
}

interface EvaluationResult {
    status: string;
    job_id: number;
    model_used: string;
    eval_source: string;
    eval_mode: string;
    evaluated_frames_count: number;
    iou_thresh: number;
    conf_thresh: number;
    overall: {
        total_gt: number;
        total_pred: number;
        total_tp: number;
        total_fp: number;
        total_fn: number;
        precision: number;
        recall: number;
        f1_score: number;
        class_miou: number;
        loc_miou: number;
        map50: number;
    };
    per_class: {
        class_name: string;
        gt: number;
        pred: number;
        tp: number;
        fp: number;
        fn: number;
        precision: number;
        recall: number;
        f1_score: number;
        miou: number;
        ap50: number;
    }[];
    confusion_pairs: {
        gt_class: string;
        pred_class: string;
        count: number;
        avg_iou: number;
    }[];
    markdown_table: string;
}

interface GtSetSummary {
    name: string;
    image_count: number;
    box_count: number;
    class_counts: Record<string, number>;
    created_at: string;
    updated_at: string;
}

interface GtSetItemDetail {
    id: string;
    source_job_id: number;
    source_frame: number;
    box_count: number;
    class_summary: Record<string, number>;
    saved_at: string;
    thumbnail: string;
}

export default function TemporalRangeModal(props: Props): JSX.Element {
    const { jobInstance, frameNumber } = props;
    const [visible, setVisible] = useState(false);
    const [loading, setLoading] = useState(false);
    const [predictLoading, setPredictLoading] = useState(false);
    const [predictAllLoading, setPredictAllLoading] = useState(false);
    const [predictAllModalVisible, setPredictAllModalVisible] = useState(false);
    const [predictAllThreshold, setPredictAllThreshold] = useState(0.15);

    const minF = jobInstance ? jobInstance.startFrame : 0;
    const maxF = jobInstance ? jobInstance.stopFrame : 100;

    // 1. 모델 관리 모달 및 상태값
    const [modelModalVisible, setModelModalVisible] = useState(false);
    const [modelsList, setModelsList] = useState<ModelItem[]>([]);
    const [activeModel, setActiveModel] = useState<string>('IR8_model');
    const [selectedTargetModel, setSelectedTargetModel] = useState<string>('IR8_model');
    const [newModelName, setNewModelName] = useState<string>('');
    const [newModelBase, setNewModelBase] = useState<string>('IR8_model');
    const [isCreatingNewModel, setIsCreatingNewModel] = useState(false);
    const [trainEpochs, setTrainEpochs] = useState<number>(5);
    const [trainLoading, setTrainLoading] = useState(false);

    // 2. 클래스 제어 모달 및 상태값 (범위 필터 지원)
    const [classModalVisible, setClassModalVisible] = useState(false);
    const [availableLabels, setAvailableLabels] = useState<LabelInfo[]>([]);

    // 탭 1: 병합 (Merge)
    const [mergeSourceId, setMergeSourceId] = useState<number | undefined>(undefined);
    const [mergeTargetId, setMergeTargetId] = useState<number | undefined>(undefined);
    const [mergeRangeMode, setMergeRangeMode] = useState<'all' | 'range'>('all');
    const [mergeStartFrame, setMergeStartFrame] = useState<number>(minF);
    const [mergeStopFrame, setMergeStopFrame] = useState<number>(maxF);
    const [mergeLoading, setMergeLoading] = useState(false);

    // 탭 2: 분리 (Split)
    const [splitInspectLabelId, setSplitInspectLabelId] = useState<number | undefined>(undefined);
    const [splitRangeMode, setSplitRangeMode] = useState<'all' | 'range'>('all');
    const [splitStartFrame, setSplitStartFrame] = useState<number>(minF);
    const [splitStopFrame, setSplitStopFrame] = useState<number>(maxF);
    const [galleryShapes, setGalleryShapes] = useState<ShapeItem[]>([]);
    const [selectedShapeIds, setSelectedShapeIds] = useState<number[]>([]);
    const [splitTargetOption, setSplitTargetOption] = useState<'existing' | 'new'>('existing');
    const [splitTargetLabelId, setSplitTargetLabelId] = useState<number | undefined>(undefined);
    const [splitNewName, setSplitNewName] = useState<string>('');
    const [splitLoading, setSplitLoading] = useState(false);

    // 탭 3: 일괄 삭제 (Delete)
    const [deleteLabelId, setDeleteLabelId] = useState<number | undefined>(undefined);
    const [deleteRangeMode, setDeleteRangeMode] = useState<'all' | 'range'>('all');
    const [deleteStartFrame, setDeleteStartFrame] = useState<number>(minF);
    const [deleteStopFrame, setDeleteStopFrame] = useState<number>(maxF);
    const [deleteLabelEntity, setDeleteLabelEntity] = useState<boolean>(false);
    const [deleteLoading, setDeleteLoading] = useState(false);

    // 3. 비지도 패턴 발견(Unsupervised Discovery) 상태값
    const [unsupervisedModalVisible, setUnsupervisedModalVisible] = useState(false);
    const [unsupervisedStart, setUnsupervisedStart] = useState<number>(minF);
    const [unsupervisedStop, setUnsupervisedStop] = useState<number>(Math.min(minF + 50, maxF));
    const [unsupervisedClusters, setUnsupervisedClusters] = useState<number>(4);
    const [unsupervisedLoading, setUnsupervisedLoading] = useState(false);

    // 4. 시계열 추적 상태값
    const [mode, setMode] = useState<string>('range_project');
    const [refStart, setRefStart] = useState<number>(Math.max(0, frameNumber - 10));
    const [refEnd, setRefEnd] = useState<number>(frameNumber);
    const [targetStart, setTargetStart] = useState<number>(frameNumber + 1);
    const [targetEnd, setTargetEnd] = useState<number>(Math.min(frameNumber + 30, maxF));
    const [n1, setN1] = useState<number>(frameNumber);
    const [n2, setN2] = useState<number>(Math.min(frameNumber + 30, maxF));
    const [threshold, setThreshold] = useState<number>(0.25);
    const [trackingEngine, setTrackingEngine] = useState<'hybrid' | 'sam'>('hybrid');

    // 5. 정확도 및 IoU 정량 평가 상태값
    const [evalModalVisible, setEvalModalVisible] = useState(false);
    const [evalModalTab, setEvalModalTab] = useState<'run' | 'gt_manage'>('run');
    const [evalLoading, setEvalLoading] = useState(false);
    const [evalSource, setEvalSource] = useState<'model' | 'annotations'>('model');
    const [evalTargetModel, setEvalTargetModel] = useState<string>(activeModel || 'IR8_model');
    const [evalMode, setEvalMode] = useState<'manual' | 'range' | 'all' | 'gt_set'>('manual');
    const [evalStartFrame, setEvalStartFrame] = useState<number>(minF);
    const [evalStopFrame, setEvalStopFrame] = useState<number>(maxF);
    const [evalIouThreshold, setEvalIouThreshold] = useState<number>(0.5);
    const [evalConfThreshold, setEvalConfThreshold] = useState<number>(0.25);
    const [evalMaxFrames, setEvalMaxFrames] = useState<number>(50);
    const [evalResult, setEvalResult] = useState<EvaluationResult | null>(null);

    // 6. 평가 전용 정답지(GT) 이미지셋 독립 관리 상태값
    const [gtSetsList, setGtSetsList] = useState<GtSetSummary[]>([]);
    const [gtSetsLoading, setGtSetsLoading] = useState(false);
    const [selectedGtSet, setSelectedGtSet] = useState<string>('');
    const [gtSaveSetName, setGtSaveSetName] = useState<string>('benchmark_gt');
    const [gtSaveFrameMode, setGtSaveFrameMode] = useState<'manual' | 'current' | 'range'>('manual');
    const [gtSaveStartFrame, setGtSaveStartFrame] = useState<number>(minF);
    const [gtSaveStopFrame, setGtSaveStopFrame] = useState<number>(maxF);
    const [gtSaveLoading, setGtSaveLoading] = useState(false);
    const [gtDetailSetName, setGtDetailSetName] = useState<string>('');
    const [gtDetailItems, setGtDetailItems] = useState<GtSetItemDetail[]>([]);
    const [gtDetailLoading, setGtDetailLoading] = useState(false);

    // 모델 목록 가져오기
    const fetchModels = async (): Promise<void> => {
        if (!jobInstance) return;
        try {
            const response = await fetch(`/api/jobs/${jobInstance.id}/models`);
            if (response.ok) {
                const data = await response.json();
                if (data.models) {
                    setModelsList(data.models);
                }
                if (data.active_model) {
                    setActiveModel(data.active_model);
                    setSelectedTargetModel(data.active_model);
                }
            }
        } catch (e) {
            console.error('Failed to fetch models', e);
        }
    };

    // 태스크 라벨 목록 가져오기
    const fetchLabels = async (): Promise<void> => {
        if (!jobInstance) return;
        try {
            if (jobInstance.labels && jobInstance.labels.length > 0) {
                const mapped = jobInstance.labels.map((l: any) => ({
                    id: l.id,
                    name: l.name,
                    color: l.color,
                }));
                setAvailableLabels(mapped);
                if (mapped.length > 0) {
                    setMergeSourceId(mapped[0].id);
                    setMergeTargetId(mapped.length > 1 ? mapped[1].id : mapped[0].id);
                    setSplitInspectLabelId(mapped[0].id);
                    setSplitTargetLabelId(mapped.length > 1 ? mapped[1].id : mapped[0].id);
                    setDeleteLabelId(mapped[0].id);
                }
            }
        } catch (e) {
            console.error('Failed to fetch labels', e);
        }
    };

    // 분리 대상 라벨의 박스 목록 조회 (범위 필터 지원)
    const fetchGallery = async (labelId?: number): Promise<void> => {
        if (!jobInstance) return;
        try {
            const params = new URLSearchParams();
            if (labelId !== undefined) params.append('label_id', String(labelId));
            if (splitRangeMode === 'range') {
                params.append('start_frame', String(splitStartFrame));
                params.append('stop_frame', String(splitStopFrame));
            }
            const url = `/api/jobs/${jobInstance.id}/labels-gallery?${params.toString()}`;
            const response = await fetch(url);
            if (response.ok) {
                const data = await response.json();
                setGalleryShapes(data.shapes || []);
                setSelectedShapeIds([]);
            }
        } catch (e) {
            console.error('Failed to fetch gallery', e);
        }
    };

    useEffect(() => {
        fetchModels();
        fetchLabels();
    }, [jobInstance?.id]);

    useEffect(() => {
        if (classModalVisible && splitInspectLabelId !== undefined) {
            fetchGallery(splitInspectLabelId);
        }
    }, [classModalVisible, splitInspectLabelId, splitRangeMode, splitStartFrame, splitStopFrame]);

    // 활성 모델 전환
    const onSelectModel = async (modelName: string): Promise<void> => {
        if (!jobInstance) return;
        try {
            const csrfToken = (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] || '';
            const response = await fetch(`/api/jobs/${jobInstance.id}/models-select`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                },
                body: JSON.stringify({ model_name: modelName }),
            });
            const data = await response.json();
            if (response.ok && data.status === 'success') {
                setActiveModel(modelName);
                setSelectedTargetModel(modelName);
                notification.success({
                    message: '활성 모델 전환 완료',
                    description: `현재 활성 모델이 '${modelName}'(으)로 변경되었습니다.`,
                });
                fetchModels();
            } else {
                notification.error({
                    message: '모델 변경 실패',
                    description: data.message || data.error || '오류가 발생했습니다.',
                });
            }
        } catch (e: any) {
            notification.error({ message: '서버 오류', description: e.message });
        }
    };

    // 새 모델 생성
    const onCreateNewModel = async (): Promise<void> => {
        if (!jobInstance || !newModelName.trim()) {
            notification.warning({ message: '모델 이름을 입력해주세요.' });
            return;
        }
        try {
            const csrfToken = (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] || '';
            const response = await fetch(`/api/jobs/${jobInstance.id}/models-create`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                },
                body: JSON.stringify({ model_name: newModelName.trim(), base_model: newModelBase }),
            });
            const data = await response.json();
            if (response.ok && data.status === 'success') {
                notification.success({
                    message: '새 모델 생성 완료!',
                    description: `새 모델 '${newModelName}'이(가) '${newModelBase}' 기반으로 생성되어 활성화되었습니다.`,
                });
                setNewModelName('');
                setIsCreatingNewModel(false);
                fetchModels();
            } else {
                notification.error({
                    message: '모델 생성 실패',
                    description: data.message || data.error || '오류가 발생했습니다.',
                });
            }
        } catch (e: any) {
            notification.error({ message: '서버 오류', description: e.message });
        }
    };

    // 모델 수동 학습 실행
    const onRunTrainModel = async (): Promise<void> => {
        if (!jobInstance) return;
        setTrainLoading(true);
        try {
            const csrfToken = (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] || '';
            const response = await fetch(`/api/jobs/${jobInstance.id}/models-train`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                },
                body: JSON.stringify({ model_name: selectedTargetModel, epochs: trainEpochs }),
            });
            const data = await response.json();
            if (response.ok && (data.status === 'training_started' || data.status === 'success')) {
                notification.success({
                    message: '🚀 모델 학습이 시작되었습니다!',
                    description: `'${selectedTargetModel}' 모델에 대해 ${trainEpochs} Epochs 실시간 학습이 백그라운드에서 시작되었습니다.`,
                    duration: 6,
                });
                fetchModels();
            } else {
                notification.error({
                    message: '학습 요청 실패',
                    description: data.message || data.error || '오류가 발생했습니다.',
                });
            }
        } catch (e: any) {
            notification.error({ message: '서버 연결 오류', description: e.message });
        } finally {
            setTrainLoading(false);
        }
    };

    // ============================================================
    // 🔄 클래스 일괄 병합 (Merge) - 범위 지원
    // ============================================================
    const onRunMergeLabels = async (): Promise<void> => {
        if (!jobInstance || !mergeSourceId || !mergeTargetId) return;
        if (mergeSourceId === mergeTargetId) {
            notification.warning({ message: '출발 클래스와 도착 클래스가 동일합니다.' });
            return;
        }
        setMergeLoading(true);
        try {
            const csrfToken = (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] || '';
            const payload: any = {
                source_label_id: mergeSourceId,
                target_label_id: mergeTargetId,
            };
            if (mergeRangeMode === 'range') {
                payload.start_frame = mergeStartFrame;
                payload.stop_frame = mergeStopFrame;
            }

            const response = await fetch(`/api/jobs/${jobInstance.id}/labels-merge`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (response.ok && data.status === 'success') {
                notification.success({
                    message: '클래스 일괄 병합 완료!',
                    description: data.message || `총 ${data.modified_count}개의 객체 라벨이 성공적으로 병합되었습니다.`,
                });
                setClassModalVisible(false);
                window.location.reload();
            } else {
                notification.error({
                    message: '병합 실패',
                    description: data.message || data.error || '오류가 발생했습니다.',
                });
            }
        } catch (e: any) {
            notification.error({ message: '서버 오류', description: e.message });
        } finally {
            setMergeLoading(false);
        }
    };

    // ============================================================
    // ✂️ 클래스 분리 및 이동 (Split)
    // ============================================================
    const onRunSplitLabels = async (): Promise<void> => {
        if (!jobInstance || selectedShapeIds.length === 0) {
            notification.warning({ message: '분리/이동할 박스를 1개 이상 선택해주세요.' });
            return;
        }
        if (splitTargetOption === 'new' && !splitNewName.trim()) {
            notification.warning({ message: '새로운 클래스 이름을 입력해주세요.' });
            return;
        }

        setSplitLoading(true);
        try {
            const csrfToken = (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] || '';
            const payload: any = {
                shape_ids: selectedShapeIds,
            };
            if (splitTargetOption === 'new') {
                payload.new_label_name = splitNewName.trim();
            } else {
                payload.target_label_id = splitTargetLabelId;
            }

            const response = await fetch(`/api/jobs/${jobInstance.id}/labels-split`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (response.ok && data.status === 'success') {
                notification.success({
                    message: '클래스 분리/이동 완료!',
                    description: `선택된 ${data.modified_count}개의 객체가 성공적으로 분리 이동되었습니다.`,
                });
                setClassModalVisible(false);
                window.location.reload();
            } else {
                notification.error({
                    message: '분리 실패',
                    description: data.message || data.error || '오류가 발생했습니다.',
                });
            }
        } catch (e: any) {
            notification.error({ message: '서버 오류', description: e.message });
        } finally {
            setSplitLoading(false);
        }
    };

    // ============================================================
    // 🗑️ 클래스 일괄 삭제 (Batch Delete) - 범위 지원
    // ============================================================
    const onRunDeleteLabels = async (): Promise<void> => {
        if (!jobInstance || !deleteLabelId) {
            notification.warning({ message: '삭제할 클래스를 선택해주세요.' });
            return;
        }
        const targetLabel = availableLabels.find((l) => l.id === deleteLabelId);
        const labelName = targetLabel ? targetLabel.name : `ID #${deleteLabelId}`;
        const rangeText = deleteRangeMode === 'range' ? ` (프레임 ${deleteStartFrame}~${deleteStopFrame})` : ' (전체 프레임)';
        if (!window.confirm(`⚠️ 정말로 '${labelName}' 클래스의 박스를${rangeText} 일괄 영구 삭제하시겠습니까?`)) {
            return;
        }

        setDeleteLoading(true);
        try {
            const csrfToken = (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] || '';
            const payload: any = {
                label_id: deleteLabelId,
                delete_label_entity: (deleteRangeMode === 'all') ? deleteLabelEntity : false,
            };
            if (deleteRangeMode === 'range') {
                payload.start_frame = deleteStartFrame;
                payload.stop_frame = deleteStopFrame;
            }

            const response = await fetch(`/api/jobs/${jobInstance.id}/labels-delete`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (response.ok && data.status === 'success') {
                notification.success({
                    message: '클래스 일괄 삭제 완료!',
                    description: data.message || `총 ${data.deleted_count}개의 박스가 성공적으로 삭제되었습니다.`,
                });
                setClassModalVisible(false);
                window.location.reload();
            } else {
                notification.error({
                    message: '삭제 실패',
                    description: data.message || data.error || '오류가 발생했습니다.',
                });
            }
        } catch (e: any) {
            notification.error({ message: '서버 오류', description: e.message });
        } finally {
            setDeleteLoading(false);
        }
    };

    // ============================================================
    // 🔮 비지도 패턴 발견 (Unsupervised Discovery)
    // ============================================================
    const onRunUnsupervisedDiscovery = async (): Promise<void> => {
        if (!jobInstance) return;
        setUnsupervisedLoading(true);
        try {
            const csrfToken = (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] || '';
            const response = await fetch(`/api/jobs/${jobInstance.id}/unsupervised-discover`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                },
                body: JSON.stringify({
                    start_frame: unsupervisedStart,
                    stop_frame: unsupervisedStop,
                    num_clusters: unsupervisedClusters,
                }),
            });
            const data = await response.json();
            if (response.ok && data.status === 'success') {
                notification.success({
                    message: '🔮 비지도 패턴 발견 완료!',
                    description: `총 ${data.created_boxes}개의 객체가 발견되어 ${data.num_clusters}개 패턴 클래스로 자동 주입되었습니다! [🏷️ 클래스] 도구에서 이름을 지정해주세요.`,
                    duration: 8,
                });
                setUnsupervisedModalVisible(false);
                window.location.reload();
            } else {
                notification.error({
                    message: '비지도 패턴 발견 실패',
                    description: data.message || data.error || '오류가 발생했습니다.',
                });
            }
        } catch (e: any) {
            notification.error({ message: '서버 연결 오류', description: e.message });
        } finally {
            setUnsupervisedLoading(false);
        }
    };

    // 1. 단일 프레임 AI 감지
    const onRunPredict = async (): Promise<void> => {
        if (!jobInstance) return;
        setPredictLoading(true);
        try {
            const csrfToken = (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] || '';
            const response = await fetch(`/api/jobs/${jobInstance.id}/ai-predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                },
                body: JSON.stringify({ frame: frameNumber, threshold: 0.15 }),
            });
            const data = await response.json();
            if (response.ok && data.status === 'success') {
                notification.success({
                    message: 'AI 자동 감지 완료!',
                    description: `프레임 #${frameNumber}에 ${data.detected_count}개의 객체가 감지되었습니다. (모델: ${data.weights_used})`,
                });
                window.location.reload();
            } else if (response.ok && data.status === 'skipped') {
                notification.info({
                    message: '기존 수작업 라벨 보호됨',
                    description: data.message || `프레임 #${frameNumber}에 이미 라벨링이 존재하여 안전하게 보존되었습니다.`,
                });
            } else {
                notification.error({
                    message: 'AI 감지 실패',
                    description: data.message || data.error || '오류가 발생했습니다.',
                });
            }
        } catch (e: any) {
            notification.error({ message: '서버 연결 오류', description: e.message });
        } finally {
            setPredictLoading(false);
        }
    };

    // 2. 전체 미라벨링 빈 프레임 일괄 감지
    const onRunPredictAll = async (): Promise<void> => {
        if (!jobInstance) return;
        setPredictAllLoading(true);
        try {
            const csrfToken = (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] || '';
            const response = await fetch(`/api/jobs/${jobInstance.id}/ai-predict-all`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                },
                body: JSON.stringify({ threshold: predictAllThreshold }),
            });
            const data = await response.json();
            if (response.ok && data.status === 'success') {
                notification.success({
                    message: '전체 미라벨링 자동 예측 완료!',
                    description: `총 ${data.empty_frames_count}개 빈 프레임에 ${data.created_boxes}개의 박스가 일괄 생성되었습니다. (모델: ${data.weights_used})`,
                });
                setPredictAllModalVisible(false);
                window.location.reload();
            } else {
                notification.error({
                    message: '일괄 예측 실패',
                    description: data.message || data.error || '오류가 발생했습니다.',
                });
            }
        } catch (e: any) {
            notification.error({ message: '서버 연결 오류', description: e.message });
        } finally {
            setPredictAllLoading(false);
        }
    };

    // 3. 시계열 연속 예측 & 양방향 보간
    const onRunTemporal = async (): Promise<void> => {
        if (!jobInstance) return;
        setLoading(true);
        try {
            const csrfToken = (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] || '';
            const payload: any = { mode, threshold, engine: trackingEngine };

            if (mode === 'range_project') {
                payload.ref_start = refStart;
                payload.ref_end = refEnd;
                payload.target_start = targetStart;
                payload.target_end = targetEnd;
            } else if (mode === 'interpolate') {
                payload.n1 = n1;
                payload.n2 = n2;
            } else {
                payload.n1 = n1;
                payload.n2 = n2;
            }

            const response = await fetch(`/api/jobs/${jobInstance.id}/temporal-range`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (response.ok && data.status === 'success') {
                notification.success({
                    message: '시계열 자동 라벨링 완료!',
                    description: `총 ${data.created_boxes}개의 시계열 추적 박스가 성공적으로 생성되었습니다. (모델: ${data.weights_used})`,
                });
                setVisible(false);
                window.location.reload();
            } else {
                notification.error({
                    message: '시계열 처리 실패',
                    description: data.message || data.error || '오류가 발생했습니다.',
                });
            }
        } catch (e: any) {
            notification.error({ message: '서버 연결 오류', description: e.message });
        } finally {
            setLoading(false);
        }
    };

    // 4. 평가 전용 정답지(GT) 이미지셋 API 호출 헬퍼
    const callEvalGtApi = async (extraPayload: Record<string, any>): Promise<any> => {
        if (!jobInstance) return null;
        const csrfToken = (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] || '';
        const response = await fetch(`/api/jobs/${jobInstance.id}/evaluate-accuracy`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,
            },
            body: JSON.stringify(extraPayload),
        });
        const data = await response.json();
        return { ok: response.ok, data };
    };

    const fetchGtSets = async (): Promise<void> => {
        if (!jobInstance) return;
        setGtSetsLoading(true);
        try {
            const res = await callEvalGtApi({ gt_action: 'list_gt_sets' });
            if (res && res.ok && res.data.status === 'success') {
                const sets: GtSetSummary[] = res.data.gt_sets || [];
                setGtSetsList(sets);
                if (sets.length > 0 && !sets.some((s) => s.name === selectedGtSet)) {
                    setSelectedGtSet(sets[0].name);
                } else if (sets.length === 0) {
                    setSelectedGtSet('');
                }
            }
        } catch (e) {
            // ignore silent refresh error
        } finally {
            setGtSetsLoading(false);
        }
    };

    useEffect(() => {
        if (evalModalVisible) {
            fetchGtSets();
        }
    }, [evalModalVisible]);

    const onLoadGtSetDetail = async (setName: string): Promise<void> => {
        setGtDetailLoading(true);
        setGtDetailSetName(setName);
        try {
            const res = await callEvalGtApi({ gt_action: 'get_gt_set_detail', set_name: setName });
            if (res && res.ok && res.data.status === 'success') {
                setGtDetailItems(res.data.items || []);
            } else {
                notification.error({ message: '정답지 상세 조회 실패', description: res?.data?.error || '오류 발생' });
            }
        } catch (e: any) {
            notification.error({ message: '서버 연결 오류', description: e.message });
        } finally {
            setGtDetailLoading(false);
        }
    };

    const onSaveToGtSet = async (): Promise<void> => {
        if (!gtSaveSetName.trim()) {
            notification.warning({ message: '정답지 셋 이름을 입력하세요.' });
            return;
        }
        setGtSaveLoading(true);
        try {
            const res = await callEvalGtApi({
                gt_action: 'save_gt_set',
                set_name: gtSaveSetName.trim(),
                frame_mode: gtSaveFrameMode,
                start_frame: gtSaveStartFrame,
                stop_frame: gtSaveStopFrame,
                current_frame: frameNumber,
            });
            if (res && res.ok && res.data.status === 'success') {
                notification.success({
                    message: '평가 전용 정답지(GT) 저장 완료',
                    description: `'${res.data.set_name}'에 ${res.data.saved_images}장 저장 (누적 ${res.data.total_images}장 / 박스 ${res.data.total_boxes}개). 학습셋과는 완전 분리 보관됩니다.`,
                    duration: 5,
                });
                setSelectedGtSet(res.data.set_name);
                await fetchGtSets();
                if (gtDetailSetName === res.data.set_name) {
                    await onLoadGtSetDetail(res.data.set_name);
                }
            } else {
                notification.error({ message: '정답지 저장 실패', description: res?.data?.error || '저장 중 오류가 발생했습니다.' });
            }
        } catch (e: any) {
            notification.error({ message: '서버 연결 오류', description: e.message });
        } finally {
            setGtSaveLoading(false);
        }
    };

    const onDeleteGtItem = async (setName: string, itemId: string): Promise<void> => {
        try {
            const res = await callEvalGtApi({ gt_action: 'delete_gt_item', set_name: setName, item_id: itemId });
            if (res && res.ok && res.data.status === 'success') {
                notification.success({ message: '정답지 이미지 제외 완료', description: `${itemId} 삭제 (남은 이미지 ${res.data.remaining_images}장)` });
                await fetchGtSets();
                await onLoadGtSetDetail(setName);
            } else {
                notification.error({ message: '삭제 실패', description: res?.data?.error || '오류 발생' });
            }
        } catch (e: any) {
            notification.error({ message: '서버 연결 오류', description: e.message });
        }
    };

    const onDeleteGtSet = async (setName: string): Promise<void> => {
        try {
            const res = await callEvalGtApi({ gt_action: 'delete_gt_set', set_name: setName });
            if (res && res.ok && res.data.status === 'success') {
                notification.success({ message: '정답지 셋 삭제 완료', description: `'${setName}'이 삭제되었습니다.` });
                if (gtDetailSetName === setName) {
                    setGtDetailSetName('');
                    setGtDetailItems([]);
                }
                await fetchGtSets();
            } else {
                notification.error({ message: '삭제 실패', description: res?.data?.error || '오류 발생' });
            }
        } catch (e: any) {
            notification.error({ message: '서버 연결 오류', description: e.message });
        }
    };

    // 4-2. 모델 정확도 및 IoU 정량 평가 실행
    const onRunEvaluation = async (): Promise<void> => {
        if (!jobInstance) return;
        if (evalMode === 'gt_set' && !selectedGtSet) {
            notification.warning({ message: '평가할 정답지(GT) 셋을 먼저 선택하거나 생성하세요.' });
            return;
        }
        setEvalLoading(true);
        try {
            const csrfToken = (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] || '';
            const payload: any = evalMode === 'gt_set' ? {
                gt_action: 'evaluate_gt_set',
                set_name: selectedGtSet,
                model_name: evalTargetModel,
                iou_threshold: evalIouThreshold,
                conf_threshold: evalConfThreshold,
                max_frames: evalMaxFrames,
            } : {
                model_name: evalTargetModel,
                eval_source: evalSource,
                eval_mode: evalMode,
                start_frame: evalStartFrame,
                stop_frame: evalStopFrame,
                iou_threshold: evalIouThreshold,
                conf_threshold: evalConfThreshold,
                max_frames: evalMaxFrames,
            };

            const response = await fetch(`/api/jobs/${jobInstance.id}/evaluate-accuracy`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (response.ok && data.status === 'success') {
                setEvalResult(data);
                notification.success({
                    message: '정확도 정량 평가 완료',
                    description: `총 ${data.evaluated_frames_count}개 프레임 평가 완료 (mIoU: ${data.overall.class_miou}%, 위치IoU: ${data.overall.loc_miou}%, mAP@0.5: ${data.overall.map50}%)`,
                    duration: 5,
                });
            } else {
                notification.error({
                    message: '정확도 평가 실패',
                    description: data.error || data.message || '평가 도중 오류가 발생했습니다.',
                });
            }
        } catch (e: any) {
            notification.error({
                message: '서버 연결 오류',
                description: e.message,
            });
        } finally {
            setEvalLoading(false);
        }
    };

    const copyMarkdownTable = (): void => {
        if (!evalResult || !evalResult.markdown_table) return;
        navigator.clipboard.writeText(evalResult.markdown_table).then(() => {
            notification.success({
                message: '보고서용 마크다운 표 복사 완료',
                description: '클립보드에 표가 복사되었습니다. 연구보고서(보고서.md)에 바로 붙여넣기(Ctrl+V) 하실 수 있습니다.',
                duration: 4,
            });
        }).catch((err) => {
            notification.error({
                message: '복사 실패',
                description: err.message,
            });
        });
    };

    return (
        <>
            {/* 1. 모델 제어 센터 버튼 */}
            <Tooltip title={`🎯 AI 모델 제어 센터 (현재 활성: ${activeModel})`}>
                <Button
                    type='link'
                    className='cvat-annotation-header-button cvat-ai-top-button'
                    style={{ color: '#722ed1' }}
                    onClick={() => {
                        fetchModels();
                        setModelModalVisible(true);
                    }}
                >
                    <SettingFilled />
                    <span>{activeModel}</span>
                </Button>
            </Tooltip>

            {/* 2. 클래스 병합/분리/삭제 관리 버튼 */}
            <Tooltip title='🏷️ 클래스 일괄 병합 & 수동 분리 & 일괄 삭제 도구'>
                <Button
                    type='link'
                    className='cvat-annotation-header-button cvat-ai-top-button'
                    style={{ color: '#13c2c2' }}
                    onClick={() => {
                        fetchLabels();
                        setClassModalVisible(true);
                    }}
                >
                    <TagsFilled />
                    <span>클래스</span>
                </Button>
            </Tooltip>

            {/* 3. 비지도 패턴 자동 발견 버튼 */}
            <Tooltip title='🔮 비지도 패턴 발견: 데이터셋에서 자주 반복되는 열화상 부품 패턴 자동 검출 및 가명 라벨링'>
                <Button
                    type='link'
                    className='cvat-annotation-header-button cvat-ai-top-button'
                    style={{ color: '#eb2f96' }}
                    onClick={() => {
                        setUnsupervisedStart(minF);
                        setUnsupervisedStop(Math.min(minF + 50, maxF));
                        setUnsupervisedModalVisible(true);
                    }}
                >
                    <BulbFilled />
                    <span>비지도</span>
                </Button>
            </Tooltip>

            {/* 4. AI 자동 감지 버튼 */}
            <Tooltip title='🤖 AI 자동 감지: 현재 프레임 객체 즉시 인식'>
                <Button
                    type='link'
                    className='cvat-annotation-header-button cvat-ai-top-button'
                    style={{ color: '#1890ff' }}
                    loading={predictLoading}
                    onClick={onRunPredict}
                >
                    <RobotFilled />
                    <span>AI 감지</span>
                </Button>
            </Tooltip>

            {/* 5. 전체 빈 프레임 일괄 감지 버튼 */}
            <Tooltip title='🚀 전체 빈 프레임 일괄 감지: 미라벨링 프레임 전체 자동 완성'>
                <Button
                    type='link'
                    className='cvat-annotation-header-button cvat-ai-top-button'
                    style={{ color: '#52c41a' }}
                    onClick={() => setPredictAllModalVisible(true)}
                >
                    <RocketFilled />
                    <span>일괄 감지</span>
                </Button>
            </Tooltip>

            {/* 6. 시계열 자동 라벨링 버튼 */}
            <Tooltip title='⚡ 시계열 자동 라벨링: 구간 참조 / 양방향 보간 / 단방향 추적'>
                <Button
                    type='link'
                    className='cvat-annotation-header-button cvat-ai-top-button'
                    style={{ color: '#fa8c16' }}
                    onClick={() => {
                        setRefStart(Math.max(0, frameNumber - 10));
                        setRefEnd(frameNumber);
                        setTargetStart(frameNumber + 1);
                        setTargetEnd(Math.min(frameNumber + 30, maxF));
                        setN1(frameNumber);
                        setN2(Math.min(frameNumber + 30, maxF));
                        setVisible(true);
                    }}
                >
                    <ThunderboltFilled />
                    <span>시계열</span>
                </Button>
            </Tooltip>

            {/* 7. 정확도 및 IoU 정량 평가 버튼 */}
            <Tooltip title='📊 모델 정확도 및 IoU 정량 평가 (mIoU / mAP / Precision / Recall / 오분류 분석)'>
                <Button
                    type='link'
                    className='cvat-annotation-header-button cvat-ai-top-button'
                    style={{ color: '#2f54eb' }}
                    onClick={() => {
                        fetchModels();
                        setEvalModalVisible(true);
                    }}
                >
                    <BarChartOutlined />
                    <span>정확도 평가</span>
                </Button>
            </Tooltip>

            {/* ============================================================ */}
            {/* 🔮 비지도 패턴 발견 & 자동 라벨링 모달 */}
            {/* ============================================================ */}
            <Modal
                title='🔮 비지도 객체 발견 & 패턴 자동 라벨링 (Unsupervised Discovery)'
                open={unsupervisedModalVisible}
                onCancel={() => setUnsupervisedModalVisible(false)}
                onOk={onRunUnsupervisedDiscovery}
                confirmLoading={unsupervisedLoading}
                okText='🚀 비지도 패턴 발견 실행'
                cancelText='취소'
                width={600}
                destroyOnClose
            >
                <div style={{ marginBottom: 16 }}>
                    <Text type='secondary'>
                        사전 라벨링이 전혀 없는 상태에서도, 데이터셋 내에서 <b>자주 반복 등장하는 열화상 부품 및 온도 패턴을 AI가 스스로 군집화(Clustering)</b>하여 가명 라벨(<code>발견패턴_1~N</code>)로 자동 주입합니다.
                    </Text>
                </div>

                <div style={{ background: '#fff0f6', padding: 12, borderRadius: 6, marginBottom: 16, border: '1px solid #ffadd2' }}>
                    <Text strong style={{ color: '#eb2f96' }}>
                        💡 사후 클래스 정제 안내:
                    </Text>
                    <div style={{ color: '#c41d7f', fontSize: 12, marginTop: 4 }}>
                        자동 주입 완료 후 상단 <b>[🏷️ 클래스]</b> 창을 열어 <code>발견패턴_1</code>을 실제 부품 이름(예: <b>콘덴서</b>)으로 원클릭 병합하거나, 불필요한 배경 패턴을 손쉽게 정리할 수 있습니다.
                    </div>
                </div>

                <div style={{ background: '#fafafa', padding: 12, borderRadius: 6, marginBottom: 16, border: '1px solid #e8e8e8' }}>
                    <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
                        <div style={{ flex: 1 }}>
                            <Text strong>분석 시작 프레임:</Text>
                            <InputNumber
                                style={{ width: '100%', marginTop: 4 }}
                                min={minF}
                                max={unsupervisedStop}
                                value={unsupervisedStart}
                                onChange={(val) => setUnsupervisedStart(Number(val))}
                            />
                        </div>
                        <div style={{ flex: 1 }}>
                            <Text strong>분석 종료 프레임:</Text>
                            <InputNumber
                                style={{ width: '100%', marginTop: 4 }}
                                min={unsupervisedStart}
                                max={maxF}
                                value={unsupervisedStop}
                                onChange={(val) => setUnsupervisedStop(Number(val))}
                            />
                        </div>
                    </div>

                    <div style={{ marginBottom: 4 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                            <Text strong>발견할 패턴(클러스터) 수:</Text>
                            <Text type='danger' strong>{unsupervisedClusters}개 패턴</Text>
                        </div>
                        <Slider
                            min={2}
                            max={8}
                            value={unsupervisedClusters}
                            onChange={(val) => setUnsupervisedClusters(Number(val))}
                        />
                        <Text type='secondary' style={{ fontSize: 11 }}>
                            (권장: 3~5개 - 부품 종류가 많을수록 값을 늘려주세요)
                        </Text>
                    </div>
                </div>
            </Modal>

            {/* ============================================================ */}
            {/* 🏷️ 클래스 일괄 병합 & 분리 & 삭제 제어 모달 (범위 지원) */}
            {/* ============================================================ */}
            <Modal
                title='🏷️ 클래스 일괄 병합(Merge) & 수동 분리(Split) & 일괄 삭제(Delete) 제어 센터'
                open={classModalVisible}
                onCancel={() => setClassModalVisible(false)}
                footer={[
                    <Button key='close' onClick={() => setClassModalVisible(false)}>
                        닫기
                    </Button>,
                ]}
                width={670}
                destroyOnClose
            >
                <Tabs defaultActiveKey='merge'>
                    {/* 탭 1: 클래스 일괄 병합 */}
                    <Tabs.TabPane
                        tab={
                            <span>
                                <SwapOutlined />
                                🔄 클래스 일괄 병합 (Merge)
                            </span>
                        }
                        key='merge'
                    >
                        <Alert
                            message='클래스 일괄 병합 안내'
                            description='지정한 범위 내에서 출발 클래스에 속한 모든 사각형 박스를 도착 클래스로 0.1초 만에 일괄 치환합니다.'
                            type='info'
                            showIcon
                            style={{ marginBottom: 14 }}
                        />

                        <div style={{ background: '#fafafa', padding: 14, borderRadius: 8, border: '1px solid #e8e8e8', marginBottom: 14 }}>
                            {/* 적용 범위 선택 */}
                            <div style={{ background: '#fff', padding: 10, borderRadius: 6, border: '1px solid #e8e8e8', marginBottom: 14 }}>
                                <Text strong style={{ display: 'block', marginBottom: 6 }}>적용 대상 프레임 범위:</Text>
                                <Radio.Group value={mergeRangeMode} onChange={(e) => setMergeRangeMode(e.target.value)}>
                                    <Radio value='all'>전체 프레임</Radio>
                                    <Radio value='range'>특정 범위 지정 (N ~ M)</Radio>
                                </Radio.Group>
                                {mergeRangeMode === 'range' && (
                                    <div style={{ display: 'flex', gap: 8, marginTop: 8, alignItems: 'center' }}>
                                        <Text>시작 (N):</Text>
                                        <InputNumber min={minF} max={mergeStopFrame} value={mergeStartFrame} onChange={(val) => setMergeStartFrame(Number(val))} style={{ width: 90 }} />
                                        <Text>~ 끝 (M):</Text>
                                        <InputNumber min={mergeStartFrame} max={maxF} value={mergeStopFrame} onChange={(val) => setMergeStopFrame(Number(val))} style={{ width: 90 }} />
                                    </div>
                                )}
                            </div>

                            <div style={{ marginBottom: 12 }}>
                                <Text strong style={{ color: '#cf1322', display: 'block', marginBottom: 4 }}>
                                    1. 변경할 기존 클래스 (출발 / Source):
                                </Text>
                                <Select
                                    style={{ width: '100%' }}
                                    value={mergeSourceId}
                                    onChange={(val) => setMergeSourceId(val)}
                                >
                                    {availableLabels.map((lbl) => (
                                        <Select.Option key={lbl.id} value={lbl.id}>
                                            <Tag color={lbl.color || 'blue'}>{lbl.name}</Tag> (ID: {lbl.id})
                                        </Select.Option>
                                    ))}
                                </Select>
                            </div>

                            <div style={{ textAlign: 'center', margin: '6px 0' }}>
                                <SwapOutlined style={{ fontSize: 18, color: '#1890ff' }} />
                            </div>

                            <div style={{ marginBottom: 14 }}>
                                <Text strong style={{ color: '#389e0d', display: 'block', marginBottom: 4 }}>
                                    2. 하나로 합칠 대상 클래스 (도착 / Target):
                                </Text>
                                <Select
                                    style={{ width: '100%' }}
                                    value={mergeTargetId}
                                    onChange={(val) => setMergeTargetId(val)}
                                >
                                    {availableLabels.map((lbl) => (
                                        <Select.Option key={lbl.id} value={lbl.id}>
                                            <Tag color={lbl.color || 'green'}>{lbl.name}</Tag> (ID: {lbl.id})
                                        </Select.Option>
                                    ))}
                                </Select>
                            </div>

                            <Button
                                type='primary'
                                icon={<SwapOutlined />}
                                loading={mergeLoading}
                                onClick={onRunMergeLabels}
                                style={{ width: '100%', height: 40, fontWeight: 'bold' }}
                            >
                                🔄 선택한 두 클래스 일괄 병합 실행
                            </Button>
                        </div>
                    </Tabs.TabPane>

                    {/* 탭 2: 클래스 수동 분리 */}
                    <Tabs.TabPane
                        tab={
                            <span>
                                <ScissorOutlined />
                                ✂️ 클래스 분리 및 이동 (Split)
                            </span>
                        }
                        key='split'
                    >
                        <Alert
                            message='클래스 수동 분리 안내'
                            description='특정 클래스에 섞여 들어간 박스들을 체크 선택하여, 새 클래스(또는 다른 기존 클래스)로 일괄 이동시킵니다.'
                            type='warning'
                            showIcon
                            style={{ marginBottom: 14 }}
                        />

                        <div style={{ marginBottom: 12 }}>
                            {/* 적용 범위 선택 */}
                            <div style={{ background: '#fff', padding: 10, borderRadius: 6, border: '1px solid #e8e8e8', marginBottom: 12 }}>
                                <Text strong style={{ display: 'block', marginBottom: 6 }}>조회 대상 프레임 범위:</Text>
                                <Radio.Group value={splitRangeMode} onChange={(e) => setSplitRangeMode(e.target.value)}>
                                    <Radio value='all'>전체 프레임</Radio>
                                    <Radio value='range'>특정 범위 지정 (N ~ M)</Radio>
                                </Radio.Group>
                                {splitRangeMode === 'range' && (
                                    <div style={{ display: 'flex', gap: 8, marginTop: 8, alignItems: 'center' }}>
                                        <Text>시작 (N):</Text>
                                        <InputNumber min={minF} max={splitStopFrame} value={splitStartFrame} onChange={(val) => setSplitStartFrame(Number(val))} style={{ width: 90 }} />
                                        <Text>~ 끝 (M):</Text>
                                        <InputNumber min={splitStartFrame} max={maxF} value={splitStopFrame} onChange={(val) => setSplitStopFrame(Number(val))} style={{ width: 90 }} />
                                    </div>
                                )}
                            </div>

                            <Text strong>조회할 원본 클래스 선택:</Text>
                            <Select
                                style={{ width: '100%', marginTop: 4 }}
                                value={splitInspectLabelId}
                                onChange={(val) => {
                                    setSplitInspectLabelId(val);
                                    fetchGallery(val);
                                }}
                            >
                                {availableLabels.map((lbl) => (
                                    <Select.Option key={lbl.id} value={lbl.id}>
                                        <Tag color={lbl.color || 'blue'}>{lbl.name}</Tag> (ID: {lbl.id})
                                    </Select.Option>
                                ))}
                            </Select>
                        </div>

                        {/* 박스 목록 (체크박스 다중 선택) */}
                        <div style={{ maxHeight: 200, overflowY: 'auto', border: '1px solid #e8e8e8', borderRadius: 6, padding: 8, marginBottom: 14, background: '#fff' }}>
                            {galleryShapes.length === 0 ? (
                                <div style={{ textAlign: 'center', padding: 16, color: '#999' }}>해당 조건에 일치하는 박스가 없습니다.</div>
                            ) : (
                                <div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, paddingBottom: 6, borderBottom: '1px solid #f0f0f0' }}>
                                        <Checkbox
                                            checked={selectedShapeIds.length === galleryShapes.length && galleryShapes.length > 0}
                                            indeterminate={selectedShapeIds.length > 0 && selectedShapeIds.length < galleryShapes.length}
                                            onChange={(e) => {
                                                if (e.target.checked) {
                                                    setSelectedShapeIds(galleryShapes.map((s) => s.id));
                                                } else {
                                                    setSelectedShapeIds([]);
                                                }
                                            }}
                                        >
                                            <b>전체 선택 ({galleryShapes.length}개)</b>
                                        </Checkbox>
                                        <Text type='secondary'>선택됨: {selectedShapeIds.length}개</Text>
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
                                        {galleryShapes.map((s) => (
                                            <div
                                                key={s.id}
                                                style={{
                                                    padding: 6,
                                                    border: '1px solid #f0f0f0',
                                                    borderRadius: 4,
                                                    background: selectedShapeIds.includes(s.id) ? '#e6f7ff' : '#fafafa',
                                                }}
                                            >
                                                <Checkbox
                                                    checked={selectedShapeIds.includes(s.id)}
                                                    onChange={(e) => {
                                                        if (e.target.checked) {
                                                            setSelectedShapeIds([...selectedShapeIds, s.id]);
                                                        } else {
                                                            setSelectedShapeIds(selectedShapeIds.filter((id) => id !== s.id));
                                                        }
                                                    }}
                                                >
                                                    <b>#{s.id}</b> — 프레임 {s.frame}번
                                                    <div style={{ fontSize: 11, color: '#888' }}>
                                                        좌표: [{s.points.join(', ')}]
                                                    </div>
                                                </Checkbox>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* 분리 대상 지정 */}
                        <div style={{ background: '#fafafa', padding: 12, borderRadius: 6, border: '1px solid #e8e8e8', marginBottom: 12 }}>
                            <Radio.Group
                                value={splitTargetOption}
                                onChange={(e) => setSplitTargetOption(e.target.value)}
                                style={{ marginBottom: 10 }}
                            >
                                <Radio value='existing'>기존 클래스로 이동</Radio>
                                <Radio value='new'>➕ 새로운 클래스 생성하여 이동</Radio>
                            </Radio.Group>

                            {splitTargetOption === 'existing' ? (
                                <Select
                                    style={{ width: '100%' }}
                                    value={splitTargetLabelId}
                                    onChange={(val) => setSplitTargetLabelId(val)}
                                >
                                    {availableLabels.map((lbl) => (
                                        <Select.Option key={lbl.id} value={lbl.id}>
                                            <Tag color={lbl.color || 'green'}>{lbl.name}</Tag>
                                        </Select.Option>
                                    ))}
                                </Select>
                            ) : (
                                <Input
                                    placeholder='새로운 클래스 이름 (예: 단자대, 열선 등)'
                                    value={splitNewName}
                                    onChange={(e) => setSplitNewName(e.target.value)}
                                />
                            )}
                        </div>

                        <Button
                            type='primary'
                            danger
                            icon={<ScissorOutlined />}
                            loading={splitLoading}
                            onClick={onRunSplitLabels}
                            style={{ width: '100%', height: 38, fontWeight: 'bold' }}
                        >
                            ✂️ 선택한 {selectedShapeIds.length}개 박스를 대상 클래스로 분리 이동
                        </Button>
                    </Tabs.TabPane>

                    {/* 탭 3: 클래스 일괄 삭제 */}
                    <Tabs.TabPane
                        tab={
                            <span>
                                <DeleteOutlined />
                                🗑️ 클래스 일괄 삭제 (Delete)
                            </span>
                        }
                        key='delete'
                    >
                        <Alert
                            message='클래스 일괄 삭제 주의'
                            description='선택한 클래스에 속한 모든 사각형 어노테이션 박스를 데이터셋에서 영구 삭제합니다.'
                            type='error'
                            showIcon
                            style={{ marginBottom: 14 }}
                        />

                        <div style={{ background: '#fff1f0', padding: 14, borderRadius: 8, border: '1px solid #ffa39e', marginBottom: 14 }}>
                            {/* 적용 범위 선택 */}
                            <div style={{ background: '#fff', padding: 10, borderRadius: 6, border: '1px solid #ffccc7', marginBottom: 14 }}>
                                <Text strong style={{ display: 'block', marginBottom: 6 }}>삭제 대상 프레임 범위:</Text>
                                <Radio.Group value={deleteRangeMode} onChange={(e) => setDeleteRangeMode(e.target.value)}>
                                    <Radio value='all'>전체 프레임</Radio>
                                    <Radio value='range'>특정 범위 지정 (N ~ M)</Radio>
                                </Radio.Group>
                                {deleteRangeMode === 'range' && (
                                    <div style={{ display: 'flex', gap: 8, marginTop: 8, alignItems: 'center' }}>
                                        <Text>시작 (N):</Text>
                                        <InputNumber min={minF} max={deleteStopFrame} value={deleteStartFrame} onChange={(val) => setDeleteStartFrame(Number(val))} style={{ width: 90 }} />
                                        <Text>~ 끝 (M):</Text>
                                        <InputNumber min={deleteStartFrame} max={maxF} value={deleteStopFrame} onChange={(val) => setDeleteStopFrame(Number(val))} style={{ width: 90 }} />
                                    </div>
                                )}
                            </div>

                            <div style={{ marginBottom: 14 }}>
                                <Text strong style={{ color: '#cf1322', display: 'block', marginBottom: 4 }}>
                                    삭제할 대상 클래스 선택:
                                </Text>
                                <Select
                                    style={{ width: '100%' }}
                                    value={deleteLabelId}
                                    onChange={(val) => setDeleteLabelId(val)}
                                >
                                    {availableLabels.map((lbl) => (
                                        <Select.Option key={lbl.id} value={lbl.id}>
                                            <Tag color={lbl.color || 'red'}>{lbl.name}</Tag> (ID: {lbl.id})
                                        </Select.Option>
                                    ))}
                                </Select>
                            </div>

                            {deleteRangeMode === 'all' && (
                                <div style={{ marginBottom: 14, background: '#fff', padding: 10, borderRadius: 6, border: '1px solid #ffccc7' }}>
                                    <Checkbox
                                        checked={deleteLabelEntity}
                                        onChange={(e) => setDeleteLabelEntity(e.target.checked)}
                                    >
                                        <Text strong style={{ color: '#a8071a' }}>
                                            태스크 라벨 목록에서도 이 라벨 자체를 완전히 영구 삭제
                                        </Text>
                                    </Checkbox>
                                    <div style={{ fontSize: 11, color: '#888', marginLeft: 24, marginTop: 2 }}>
                                        (체크 해제 시: 박스들만 지우고 라벨 이름은 남겨둡니다)
                                    </div>
                                </div>
                            )}

                            <Button
                                type='primary'
                                danger
                                icon={<DeleteOutlined />}
                                loading={deleteLoading}
                                onClick={onRunDeleteLabels}
                                style={{ width: '100%', height: 40, fontWeight: 'bold' }}
                            >
                                🗑️ 선택한 클래스 데이터 일괄 영구 삭제 실행
                            </Button>
                        </div>
                    </Tabs.TabPane>
                </Tabs>
            </Modal>

            {/* ============================================================ */}
            {/* 🎯 모델 관리 & 수동 학습 실행 통합 모달 */}
            {/* ============================================================ */}
            <Modal
                title='🎯 인공지능 모델 분리 관리 & 학습 제어 센터'
                open={modelModalVisible}
                onCancel={() => setModelModalVisible(false)}
                footer={[
                    <Button key='close' onClick={() => setModelModalVisible(false)}>
                        닫기
                    </Button>,
                ]}
                width={620}
                destroyOnClose
            >
                <div style={{ marginBottom: 16 }}>
                    <Text type='secondary'>
                        작업 환경(배전반, 변압기 등)에 따라 독립된 모델을 생성하고, 원하는 시점에 명시적으로 학습을 실행할 수 있습니다.
                    </Text>
                </div>

                {/* 섹션 1: 활성 모델 선택 */}
                <Card size='small' title='📌 현재 활성 모델 선택 (Active Model)' style={{ marginBottom: 16, background: '#fafafa' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
                        <div style={{ flex: 1 }}>
                            <Select
                                style={{ width: '100%' }}
                                value={selectedTargetModel}
                                onChange={(val) => setSelectedTargetModel(val)}
                            >
                                {modelsList.map((m) => (
                                    <Select.Option key={m.name} value={m.name}>
                                        <b>{m.name}</b> {m.name === activeModel ? '(현재 활성)' : ''} — {m.size_mb} MB ({m.modified})
                                    </Select.Option>
                                ))}
                            </Select>
                        </div>
                        {selectedTargetModel !== activeModel && (
                            <Button type='primary' onClick={() => onSelectModel(selectedTargetModel)}>
                                활성화 전환
                            </Button>
                        )}
                        {selectedTargetModel === activeModel && (
                            <Tag color='success' icon={<CheckCircleFilled />}>
                                현재 사용 중
                            </Tag>
                        )}
                    </div>
                </Card>

                {/* 섹션 2: 수동 학습 실행 (On-Demand Training) */}
                <Card
                    size='small'
                    title='🧠 선택된 모델 실시간 학습 실행 (On-Demand Train)'
                    style={{ marginBottom: 16, border: '1px solid #d3adf7', background: '#f9f0ff' }}
                >
                    <Text type='secondary' style={{ display: 'block', marginBottom: 10 }}>
                        지금까지 검수한 모든 열화상 프레임 데이터를 수집하여 <b>'{selectedTargetModel}'</b> 모델을 집중 훈련시킵니다.
                    </Text>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                        <div style={{ width: 140 }}>
                            <Text strong>학습 에포크 (Epochs):</Text>
                        </div>
                        <div style={{ flex: 1 }}>
                            <Slider
                                min={1}
                                max={20}
                                value={trainEpochs}
                                onChange={(val) => setTrainEpochs(Number(val))}
                            />
                        </div>
                        <InputNumber
                            min={1}
                            max={50}
                            value={trainEpochs}
                            onChange={(val) => setTrainEpochs(Number(val))}
                            style={{ width: 70 }}
                        />
                    </div>
                    <Button
                        type='primary'
                        danger
                        icon={<FireFilled />}
                        loading={trainLoading}
                        onClick={onRunTrainModel}
                        style={{ width: '100%', height: 40, fontWeight: 'bold', fontSize: 15 }}
                    >
                        🚀 '{selectedTargetModel}' 모델 실시간 학습 시작
                    </Button>
                </Card>

                {/* 섹션 3: 새 독립 모델 추가/복제 */}
                <Card size='small' title='➕ 새로운 독립 모델 추가 (Create / Clone)'>
                    {!isCreatingNewModel ? (
                        <Button
                            type='dashed'
                            icon={<PlusOutlined />}
                            onClick={() => setIsCreatingNewModel(true)}
                            style={{ width: '100%' }}
                        >
                            새 모델 생성하기
                        </Button>
                    ) : (
                        <div style={{ background: '#fff', padding: 12, borderRadius: 6, border: '1px solid #e8e8e8' }}>
                            <div style={{ marginBottom: 10 }}>
                                <Text strong>새 모델 이름:</Text>
                                <Input
                                    placeholder='예: 야외_변압기_v1, 콘덴서_특화_v2'
                                    value={newModelName}
                                    onChange={(e) => setNewModelName(e.target.value)}
                                    style={{ marginTop: 4 }}
                                />
                            </div>
                            <div style={{ marginBottom: 12 }}>
                                <Text strong>베이스 가중치 복제:</Text>
                                <Select
                                    style={{ width: '100%', marginTop: 4 }}
                                    value={newModelBase}
                                    onChange={(val) => setNewModelBase(val)}
                                >
                                    {modelsList.map((m) => (
                                        <Select.Option key={m.name} value={m.name}>
                                            {m.name} ({m.size_mb} MB)
                                        </Select.Option>
                                    ))}
                                </Select>
                            </div>
                            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                                <Button onClick={() => setIsCreatingNewModel(false)}>취소</Button>
                                <Button type='primary' onClick={onCreateNewModel}>
                                    생성 및 활성화
                                </Button>
                            </div>
                        </div>
                    )}
                </Card>
            </Modal>

            {/* 전체 빈 프레임 일괄 감지 모달 */}
            <Modal
                title='🚀 전체 미라벨링 빈 프레임 일괄 AI 예측'
                open={predictAllModalVisible}
                onCancel={() => setPredictAllModalVisible(false)}
                onOk={onRunPredictAll}
                confirmLoading={predictAllLoading}
                okText='🚀 전체 일괄 예측 실행'
                cancelText='취소'
                destroyOnClose
            >
                <div style={{ marginBottom: 16 }}>
                    <Text type='secondary'>
                        현재 활성 모델 <b>[{activeModel}]</b>(으)로 라벨링이 비어있는 모든 프레임을 한 번에 자동 완성합니다.
                    </Text>
                </div>
                <div style={{ background: '#f6ffed', padding: 12, borderRadius: 6, marginBottom: 16, border: '1px solid #b7eb8f' }}>
                    <Text strong style={{ color: '#389e0d' }}>
                        🛡️ 수작업 라벨링 100% 안전 보존:
                    </Text>
                    <div style={{ color: '#52c41a', fontSize: 12, marginTop: 4 }}>
                        이미 작업자가 라벨링해 둔 프레임은 전혀 건드리지 않고 원본 그대로 보존됩니다.
                    </div>
                </div>
                <div style={{ marginBottom: 12 }}>
                    <Text strong>신뢰도 임계값 (Threshold: {predictAllThreshold}):</Text>
                    <Slider
                        min={0.05}
                        max={0.95}
                        step={0.05}
                        value={predictAllThreshold}
                        onChange={(val) => setPredictAllThreshold(Number(val))}
                    />
                </div>
            </Modal>

            {/* 시계열 3단계 모드 모달 */}
            <Modal
                title='⚡ 열화상 시계열 인공지능 라벨링 & 보간 도구'
                open={visible}
                onCancel={() => setVisible(false)}
                onOk={onRunTemporal}
                confirmLoading={loading}
                okText='🚀 시계열 라벨링 실행'
                cancelText='취소'
                width={560}
                destroyOnClose
            >
                <div style={{ marginBottom: 16 }}>
                    <Text strong style={{ display: 'block', marginBottom: 8 }}>추적 및 보간 모드 선택:</Text>
                    <Radio.Group value={mode} onChange={(e) => setMode(e.target.value)}>
                        <Radio.Button value='range_project'>🎯 구간 패턴 참조 → 미래 예측</Radio.Button>
                        <Radio.Button value='interpolate'>🔄 양방향 보간 (N1 ↔ N2)</Radio.Button>
                        <Radio.Button value='forward'>⚡ 단방향 연속 추적</Radio.Button>
                    </Radio.Group>
                </div>

                <div style={{ background: '#f6ffed', padding: '10px 14px', borderRadius: 6, marginBottom: 16, border: '1px solid #b7eb8f' }}>
                    <Text strong style={{ display: 'block', marginBottom: 6, color: '#237804' }}>
                        🧠 추적 엔진 선택:
                    </Text>
                    <Radio.Group value={trackingEngine} onChange={(e) => setTrackingEngine(e.target.value)}>
                        <Radio value='hybrid' style={{ display: 'block', marginBottom: 4 }}>
                            <b>⚡ 하이브리드 고속 엔진</b> <span style={{ color: '#8c8c8c' }}>(YOLOv8 + 광학흐름 + 보간 / 0.1초 완료)</span>
                        </Radio>
                        <Radio value='sam' style={{ display: 'block' }}>
                            <b>🎯 SAM 비주얼 트래커</b> <span style={{ color: '#8c8c8c' }}>(Segment Anything 시각 마스크 기반 정밀 BBox 추적)</span>
                        </Radio>
                    </Radio.Group>
                </div>

                {/* 모드 1: 구간 패턴 참조 -> 미래 예측 */}
                {mode === 'range_project' && (
                    <div style={{ background: '#fafafa', padding: 12, borderRadius: 6, marginBottom: 16, border: '1px solid #e8e8e8' }}>
                        <Text type='secondary' style={{ display: 'block', marginBottom: 10 }}>
                            📌 <b>[학습/참조 구간]</b>의 수작업 라벨링 패턴을 이어받아 <b>[자동 예측 구간]</b>에 박스를 연속 생성합니다.
                        </Text>
                        <div style={{ display: 'flex', gap: 12, marginBottom: 10 }}>
                            <div style={{ flex: 1 }}>
                                <Text strong style={{ color: '#096dd9' }}>참조 시작 (A):</Text>
                                <InputNumber
                                    style={{ width: '100%', marginTop: 4 }}
                                    min={minF}
                                    max={refEnd}
                                    value={refStart}
                                    onChange={(val) => setRefStart(Number(val))}
                                />
                            </div>
                            <div style={{ flex: 1 }}>
                                <Text strong style={{ color: '#096dd9' }}>참조 끝 (B):</Text>
                                <InputNumber
                                    style={{ width: '100%', marginTop: 4 }}
                                    min={refStart}
                                    max={maxF}
                                    value={refEnd}
                                    onChange={(val) => {
                                        const v = Number(val);
                                        setRefEnd(v);
                                        if (targetStart <= v) setTargetStart(v + 1);
                                    }}
                                />
                            </div>
                        </div>
                        <div style={{ display: 'flex', gap: 12 }}>
                            <div style={{ flex: 1 }}>
                                <Text strong style={{ color: '#d46b08' }}>예측 시작 (C):</Text>
                                <InputNumber
                                    style={{ width: '100%', marginTop: 4 }}
                                    min={refEnd + 1}
                                    max={targetEnd}
                                    value={targetStart}
                                    onChange={(val) => setTargetStart(Number(val))}
                                />
                            </div>
                            <div style={{ flex: 1 }}>
                                <Text strong style={{ color: '#d46b08' }}>예측 끝 (D):</Text>
                                <InputNumber
                                    style={{ width: '100%', marginTop: 4 }}
                                    min={targetStart}
                                    max={maxF}
                                    value={targetEnd}
                                    onChange={(val) => setTargetEnd(Number(val))}
                                />
                            </div>
                        </div>
                    </div>
                )}

                {/* 모드 2: 양방향 보간 */}
                {mode === 'interpolate' && (
                    <div style={{ background: '#fafafa', padding: 12, borderRadius: 6, marginBottom: 16, border: '1px solid #e8e8e8' }}>
                        <Text type='secondary' style={{ display: 'block', marginBottom: 10 }}>
                            🔄 시작점(N1)과 끝점(N2)에 그려진 사각형 사이의 <b>중간 프레임(N1+1 ~ N2-1)을 부드럽게 보간</b>합니다. (기존 라벨 보존)
                        </Text>
                        <div style={{ display: 'flex', gap: 12 }}>
                            <div style={{ flex: 1 }}>
                                <Text strong>시작 프레임 (N1):</Text>
                                <InputNumber
                                    style={{ width: '100%', marginTop: 4 }}
                                    min={minF}
                                    max={maxF}
                                    value={n1}
                                    onChange={(val) => setN1(Number(val))}
                                />
                            </div>
                            <div style={{ flex: 1 }}>
                                <Text strong>끝 프레임 (N2):</Text>
                                <InputNumber
                                    style={{ width: '100%', marginTop: 4 }}
                                    min={n1 + 2}
                                    max={maxF}
                                    value={n2}
                                    onChange={(val) => setN2(Number(val))}
                                />
                            </div>
                        </div>
                    </div>
                )}

                {/* 모드 3: 단방향 연속 추적 */}
                {mode === 'forward' && (
                    <div style={{ background: '#fafafa', padding: 12, borderRadius: 6, marginBottom: 16, border: '1px solid #e8e8e8' }}>
                        <Text type='secondary' style={{ display: 'block', marginBottom: 10 }}>
                            ⚡ 시작 프레임(N1)의 객체를 기준으로 끝 프레임(N2)까지 연속 추적합니다. (기존 라벨 보존)
                        </Text>
                        <div style={{ display: 'flex', gap: 12 }}>
                            <div style={{ flex: 1 }}>
                                <Text strong>시작 프레임 (N1):</Text>
                                <InputNumber
                                    style={{ width: '100%', marginTop: 4 }}
                                    min={minF}
                                    max={maxF}
                                    value={n1}
                                    onChange={(val) => setN1(Number(val))}
                                />
                            </div>
                            <div style={{ flex: 1 }}>
                                <Text strong>끝 프레임 (N2):</Text>
                                <InputNumber
                                    style={{ width: '100%', marginTop: 4 }}
                                    min={n1}
                                    max={maxF}
                                    value={n2}
                                    onChange={(val) => setN2(Number(val))}
                                />
                            </div>
                        </div>
                    </div>
                )}

                <div style={{ marginBottom: 12 }}>
                    <Text strong>신뢰도 임계값 (Threshold: {threshold}):</Text>
                    <Slider
                        min={0.05}
                        max={0.95}
                        step={0.05}
                        value={threshold}
                        onChange={(val) => setThreshold(Number(val))}
                    />
                </div>
            </Modal>

            {/* ============================================================ */}
            {/* 📊 AI 탐지 정확도 및 IoU 정량 평가 모달 */}
            {/* ============================================================ */}
            <Modal
                title='📊 AI 모델 탐지 정확도 및 IoU 정량 평가 (Quantitative Evaluation)'
                open={evalModalVisible}
                onCancel={() => setEvalModalVisible(false)}
                width={980}
                footer={[
                    evalModalTab === 'run' && evalResult && (
                        <Button key='copy' type='default' icon={<CopyOutlined />} onClick={copyMarkdownTable} style={{ marginRight: 'auto' }}>
                            📋 보고서용 Markdown 표 복사
                        </Button>
                    ),
                    <Button key='close' onClick={() => setEvalModalVisible(false)}>
                        닫기
                    </Button>,
                    evalModalTab === 'run' && (
                        <Button
                            key='run'
                            type='primary'
                            icon={<BarChartOutlined />}
                            loading={evalLoading}
                            onClick={onRunEvaluation}
                            style={{ backgroundColor: '#2f54eb', borderColor: '#2f54eb' }}
                        >
                            평가 실행
                        </Button>
                    ),
                ]}
            >
                <Tabs
                    activeKey={evalModalTab}
                    onChange={(k) => setEvalModalTab(k as 'run' | 'gt_manage')}
                    items={[
                        {
                            key: 'run',
                            label: '📊 정량 평가 실행',
                            children: (
                                <>
                                    {/* 1. 평가 옵션 설정 카드 */}
                                    <Card size='small' style={{ marginBottom: 16, background: '#f9fbfd', border: '1px solid #d6e4ff' }}>
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                                            {/* 평가 대상 프레임 */}
                                            <div>
                                                <Text strong style={{ marginRight: 12 }}>🎯 평가 대상 (Ground Truth):</Text>
                                                <Radio.Group
                                                    value={evalMode}
                                                    onChange={(e) => {
                                                        const v = e.target.value;
                                                        setEvalMode(v);
                                                        if (v === 'gt_set') {
                                                            setEvalSource('model');
                                                        }
                                                    }}
                                                >
                                                    <Radio.Button value='manual'>수작업 검수 프레임만 [권장]</Radio.Button>
                                                    <Radio.Button value='range'>지정 구간 (N ~ M)</Radio.Button>
                                                    <Radio.Button value='all'>전체 프레임</Radio.Button>
                                                    <Radio.Button value='gt_set'>📁 저장된 정답지(GT) 셋</Radio.Button>
                                                </Radio.Group>
                                            </div>

                                            {evalMode === 'gt_set' && (
                                                <div style={{ display: 'flex', gap: 12, alignItems: 'center', paddingLeft: 12, flexWrap: 'wrap' }}>
                                                    <Text strong>정답지 셋 선택:</Text>
                                                    <Select
                                                        value={selectedGtSet || undefined}
                                                        placeholder='저장된 정답지 셋을 선택하세요'
                                                        onChange={(val) => setSelectedGtSet(val)}
                                                        style={{ width: 280 }}
                                                        loading={gtSetsLoading}
                                                    >
                                                        {gtSetsList.map((gs) => (
                                                            <Select.Option key={gs.name} value={gs.name}>
                                                                {gs.name} ({gs.image_count}장 / {gs.box_count}박스)
                                                            </Select.Option>
                                                        ))}
                                                    </Select>
                                                    <Button size='small' onClick={() => setEvalModalTab('gt_manage')}>
                                                        + 정답지 이미지 관리
                                                    </Button>
                                                    {gtSetsList.length === 0 && (
                                                        <Text type='secondary'>저장된 정답지 셋이 없습니다. [📁 정답용(GT) 이미지 관리] 탭에서 먼저 저장하세요.</Text>
                                                    )}
                                                </div>
                                            )}

                                            {evalMode === 'range' && (
                                                <div style={{ display: 'flex', gap: 16, alignItems: 'center', paddingLeft: 12 }}>
                                                    <div>
                                                        <Text type='secondary'>시작 프레임: </Text>
                                                        <InputNumber
                                                            min={minF}
                                                            max={maxF}
                                                            value={evalStartFrame}
                                                            onChange={(val) => setEvalStartFrame(Number(val))}
                                                            style={{ width: 90 }}
                                                        />
                                                    </div>
                                                    <div>
                                                        <Text type='secondary'>종료 프레임: </Text>
                                                        <InputNumber
                                                            min={evalStartFrame}
                                                            max={maxF}
                                                            value={evalStopFrame}
                                                            onChange={(val) => setEvalStopFrame(Number(val))}
                                                            style={{ width: 90 }}
                                                        />
                                                    </div>
                                                </div>
                                            )}

                                            {/* 평가 소스 */}
                                            <div style={{ display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap' }}>
                                                {evalMode !== 'gt_set' && (
                                                    <div>
                                                        <Text strong style={{ marginRight: 12 }}>🔍 예측 데이터 소스:</Text>
                                                        <Radio.Group
                                                            value={evalSource}
                                                            onChange={(e) => setEvalSource(e.target.value)}
                                                        >
                                                            <Radio.Button value='model'>AI 모델 직접 추론 (YOLOv8)</Radio.Button>
                                                            <Radio.Button value='annotations'>CVAT 자동 라벨 (Auto Shapes)</Radio.Button>
                                                        </Radio.Group>
                                                    </div>
                                                )}

                                                {(evalSource === 'model' || evalMode === 'gt_set') && (
                                                    <div>
                                                        <Text strong style={{ marginRight: 8 }}>평가 모델:</Text>
                                                        <Select
                                                            value={evalTargetModel}
                                                            onChange={(val) => setEvalTargetModel(val)}
                                                            style={{ width: 160 }}
                                                        >
                                                            {modelsList.map((m) => (
                                                                <Select.Option key={m.name} value={m.name}>
                                                                    {m.name} {m.is_active ? '⭐' : ''}
                                                                </Select.Option>
                                                            ))}
                                                        </Select>
                                                    </div>
                                                )}

                                                <div>
                                                    <Text strong style={{ marginRight: 8 }}>최대 평가 프레임:</Text>
                                                    <InputNumber
                                                        min={5}
                                                        max={200}
                                                        value={evalMaxFrames}
                                                        onChange={(val) => setEvalMaxFrames(Number(val))}
                                                        style={{ width: 70 }}
                                                    />
                                                </div>
                                            </div>

                                            {/* 임계값 슬라이더 */}
                                            <div style={{ display: 'flex', gap: 24 }}>
                                                <div style={{ flex: 1 }}>
                                                    <Text strong>
                                                        IoU 매칭 임계값 (IoU Threshold: {evalIouThreshold})
                                                    </Text>
                                                    <Slider
                                                        min={0.3}
                                                        max={0.9}
                                                        step={0.05}
                                                        value={evalIouThreshold}
                                                        onChange={(val) => setEvalIouThreshold(Number(val))}
                                                    />
                                                </div>
                                                {(evalSource === 'model' || evalMode === 'gt_set') && (
                                                    <div style={{ flex: 1 }}>
                                                        <Text strong>
                                                            감지 신뢰도 임계값 (Confidence: {evalConfThreshold})
                                                        </Text>
                                                        <Slider
                                                            min={0.05}
                                                            max={0.9}
                                                            step={0.05}
                                                            value={evalConfThreshold}
                                                            onChange={(val) => setEvalConfThreshold(Number(val))}
                                                        />
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </Card>

                {/* 2. 평가 결과 영역 */}
                {evalLoading && (
                    <div style={{ textAlign: 'center', padding: '40px 0' }}>
                        <Spin size='large' tip='선택된 프레임 이미지와 정답지(GT)를 기반으로 mIoU 및 mAP@0.5를 정밀 연산 중입니다...' />
                    </div>
                )}

                {!evalLoading && evalResult && evalResult.overall && (
                    <div>
                        {/* 핵심 지표 카드 행 */}
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 10, marginBottom: 16 }}>
                            <Card size='small' style={{ textAlign: 'center', background: '#e6f7ff', borderColor: '#91d5ff' }}>
                                <div style={{ fontSize: 11, color: '#0050b3' }}>클래스 mIoU</div>
                                <div style={{ fontSize: 20, fontWeight: 'bold', color: '#096dd9' }}>
                                    {evalResult.overall.class_miou}%
                                </div>
                                <div style={{ fontSize: 10, color: '#8c8c8c' }}>클래스 일치 기준</div>
                            </Card>

                            <Card size='small' style={{ textAlign: 'center', background: '#f6ffed', borderColor: '#b7eb8f' }}>
                                <div style={{ fontSize: 11, color: '#237804' }}>위치 정합도 (Loc)</div>
                                <div style={{ fontSize: 20, fontWeight: 'bold', color: '#389e0d' }}>
                                    {evalResult.overall.loc_miou}%
                                </div>
                                <div style={{ fontSize: 10, color: '#8c8c8c' }}>BBox 물리 경계</div>
                            </Card>

                            <Card size='small' style={{ textAlign: 'center', background: '#f9f0ff', borderColor: '#d3adf7' }}>
                                <div style={{ fontSize: 11, color: '#531dab' }}>mAP@0.5</div>
                                <div style={{ fontSize: 20, fontWeight: 'bold', color: '#722ed1' }}>
                                    {evalResult.overall.map50}%
                                </div>
                                <div style={{ fontSize: 10, color: '#8c8c8c' }}>PASCAL VOC 11-pt</div>
                            </Card>

                            <Card size='small' style={{ textAlign: 'center', background: '#e6fffb', borderColor: '#87e8de' }}>
                                <div style={{ fontSize: 11, color: '#006d75' }}>Precision (정밀도)</div>
                                <div style={{ fontSize: 20, fontWeight: 'bold', color: '#08979c' }}>
                                    {evalResult.overall.precision}%
                                </div>
                                <div style={{ fontSize: 10, color: '#8c8c8c' }}>TP / (TP+FP)</div>
                            </Card>

                            <Card size='small' style={{ textAlign: 'center', background: '#fff7e6', borderColor: '#ffd591' }}>
                                <div style={{ fontSize: 11, color: '#ad4e00' }}>Recall (재현율)</div>
                                <div style={{ fontSize: 20, fontWeight: 'bold', color: '#d46b08' }}>
                                    {evalResult.overall.recall}%
                                </div>
                                <div style={{ fontSize: 10, color: '#8c8c8c' }}>TP / (TP+FN)</div>
                            </Card>

                            <Card size='small' style={{ textAlign: 'center', background: '#fff0f6', borderColor: '#ffadd2' }}>
                                <div style={{ fontSize: 11, color: '#9e1068' }}>F1-Score</div>
                                <div style={{ fontSize: 20, fontWeight: 'bold', color: '#c41d7f' }}>
                                    {evalResult.overall.f1_score}
                                </div>
                                <div style={{ fontSize: 10, color: '#8c8c8c' }}>조화 평균</div>
                            </Card>
                        </div>

                        {/* 통계 요약 텍스트 */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                            <Text type='secondary'>
                                평가 프레임 수: <b>{evalResult.evaluated_frames_count}</b>개 | 총 정답지(GT): <b>{evalResult.overall.total_gt}</b>개 | 감지 박스: <b>{evalResult.overall.total_pred}</b>개 (TP: {evalResult.overall.total_tp}, FP: {evalResult.overall.total_fp}, FN: {evalResult.overall.total_fn})
                            </Text>
                            <Button size='small' type='primary' ghost icon={<CopyOutlined />} onClick={copyMarkdownTable}>
                                보고서용 표 복사
                            </Button>
                        </div>

                        {/* 오분류 및 미세 경계 패턴 안내 (있는 경우) */}
                        {evalResult.confusion_pairs && evalResult.confusion_pairs.length > 0 && (
                            <Alert
                                style={{ marginBottom: 12 }}
                                message='🔍 교차 오분류 패턴 분석 (Active Learning 추가 학습 추천)'
                                description={
                                    <ul style={{ margin: 0, paddingLeft: 16 }}>
                                        {evalResult.confusion_pairs.map((cp, idx) => (
                                            <li key={idx}>
                                                GT <b>{cp.gt_class}</b> ➔ 모델 예측 <b>{cp.pred_class}</b> : 총 <b>{cp.count}건</b> (평균 BBox 위치 일치율: <b>{cp.avg_iou}%</b>)
                                            </li>
                                        ))}
                                    </ul>
                                }
                                type='info'
                                showIcon
                            />
                        )}

                        {/* 클래스별 정량 평가 상세 테이블 */}
                        <Table
                            size='small'
                            dataSource={evalResult.per_class.map((c, i) => ({ ...c, key: i }))}
                            pagination={evalResult.per_class.length > 6 ? { pageSize: 6 } : false}
                            scroll={{ y: 220 }}
                            columns={[
                                {
                                    title: '부품 클래스',
                                    dataIndex: 'class_name',
                                    key: 'class_name',
                                    render: (val: string) => <Tag color='blue'>{val}</Tag>,
                                },
                                { title: 'GT', dataIndex: 'gt', key: 'gt', align: 'center' as const, width: 55 },
                                { title: '예측', dataIndex: 'pred', key: 'pred', align: 'center' as const, width: 55 },
                                { title: 'TP', dataIndex: 'tp', key: 'tp', align: 'center' as const, width: 50 },
                                { title: 'FP', dataIndex: 'fp', key: 'fp', align: 'center' as const, width: 50 },
                                { title: 'FN', dataIndex: 'fn', key: 'fn', align: 'center' as const, width: 50 },
                                {
                                    title: '정밀도',
                                    dataIndex: 'precision',
                                    key: 'precision',
                                    align: 'center' as const,
                                    render: (v: number) => `${v}%`,
                                },
                                {
                                    title: '재현율',
                                    dataIndex: 'recall',
                                    key: 'recall',
                                    align: 'center' as const,
                                    render: (v: number) => `${v}%`,
                                },
                                {
                                    title: 'F1',
                                    dataIndex: 'f1_score',
                                    key: 'f1_score',
                                    align: 'center' as const,
                                },
                                {
                                    title: 'mIoU',
                                    dataIndex: 'miou',
                                    key: 'miou',
                                    align: 'center' as const,
                                    render: (v: number) => <span style={{ color: v >= 50 ? '#52c41a' : '#fa8c16' }}>{v}%</span>,
                                },
                                {
                                    title: 'AP@0.5',
                                    dataIndex: 'ap50',
                                    key: 'ap50',
                                    align: 'center' as const,
                                    render: (v: number) => `${v}%`,
                                },
                            ]}
                        />
                    </div>
                )}
                                </>
                            ),
                        },
                        {
                            key: 'gt_manage',
                            label: `📁 정답용(GT) 이미지 관리 (${gtSetsList.length})`,
                            children: (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                                    <Alert
                                        type='info'
                                        showIcon
                                        message='평가 전용 정답지(GT) 독립 보관소'
                                        description='여기에 저장된 이미지와 라벨은 active_learning_dataset(학습셋)과 완전히 분리된 eval_gt_datasets 폴더에 보관되며, 모델 재학습에 섞이지 않습니다.'
                                    />

                                    {/* 1) 현재 Job의 수작업 GT를 정답지 셋으로 저장 */}
                                    <Card size='small' title='➕ 현재 Job 수작업(Manual) 라벨 → 정답지 셋에 저장/추가' style={{ background: '#f6ffed', border: '1px solid #b7eb8f' }}>
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                                            <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                                                <Text strong>정답지 셋 이름:</Text>
                                                <Input
                                                    value={gtSaveSetName}
                                                    onChange={(e) => setGtSaveSetName(e.target.value)}
                                                    placeholder='예: benchmark_gt, 현장검증_v1'
                                                    style={{ width: 220 }}
                                                />
                                                {gtSetsList.length > 0 && (
                                                    <Select
                                                        placeholder='기존 셋에 추가'
                                                        style={{ width: 170 }}
                                                        value={undefined}
                                                        onChange={(val: string) => setGtSaveSetName(val)}
                                                    >
                                                        {gtSetsList.map((gs) => (
                                                            <Select.Option key={gs.name} value={gs.name}>{gs.name}</Select.Option>
                                                        ))}
                                                    </Select>
                                                )}
                                                <Radio.Group
                                                    value={gtSaveFrameMode}
                                                    onChange={(e) => setGtSaveFrameMode(e.target.value)}
                                                    size='small'
                                                >
                                                    <Radio.Button value='manual'>전체 수작업 프레임</Radio.Button>
                                                    <Radio.Button value='current'>현재 프레임(#{frameNumber})</Radio.Button>
                                                    <Radio.Button value='range'>지정 구간</Radio.Button>
                                                </Radio.Group>
                                                <Button
                                                    type='primary'
                                                    icon={<PlusOutlined />}
                                                    loading={gtSaveLoading}
                                                    onClick={onSaveToGtSet}
                                                    style={{ backgroundColor: '#52c41a', borderColor: '#52c41a' }}
                                                >
                                                    정답지 저장
                                                </Button>
                                            </div>
                                            {gtSaveFrameMode === 'range' && (
                                                <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                                                    <Text type='secondary'>시작 프레임:</Text>
                                                    <InputNumber min={minF} max={maxF} value={gtSaveStartFrame} onChange={(v) => setGtSaveStartFrame(Number(v))} style={{ width: 90 }} />
                                                    <Text type='secondary'>종료 프레임:</Text>
                                                    <InputNumber min={gtSaveStartFrame} max={maxF} value={gtSaveStopFrame} onChange={(v) => setGtSaveStopFrame(Number(v))} style={{ width: 90 }} />
                                                </div>
                                            )}
                                        </div>
                                    </Card>

                                    {/* 2) 저장된 정답지 셋 목록 */}
                                    <Card
                                        size='small'
                                        title='📚 저장된 정답지(GT) 셋 목록'
                                        extra={(
                                            <Button size='small' icon={<ReloadOutlined />} onClick={fetchGtSets} loading={gtSetsLoading}>
                                                새로고침
                                            </Button>
                                        )}
                                    >
                                        <Table
                                            size='small'
                                            dataSource={gtSetsList}
                                            rowKey='name'
                                            loading={gtSetsLoading}
                                            pagination={false}
                                            locale={{ emptyText: '저장된 정답지 셋이 없습니다. 위에서 현재 Job의 수작업 라벨을 저장해보세요.' }}
                                            columns={[
                                                { title: '정답지 셋 이름', dataIndex: 'name', key: 'name', render: (v: string) => <Text strong>{v}</Text> },
                                                { title: '이미지 수', dataIndex: 'image_count', key: 'image_count', align: 'center' as const, render: (v: number) => <Tag color='blue'>{v}장</Tag> },
                                                { title: 'GT 박스 수', dataIndex: 'box_count', key: 'box_count', align: 'center' as const, render: (v: number) => <Tag color='purple'>{v}개</Tag> },
                                                {
                                                    title: '포함 클래스',
                                                    dataIndex: 'class_counts',
                                                    key: 'class_counts',
                                                    render: (cc: Record<string, number>) => (
                                                        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                                                            {Object.entries(cc || {}).slice(0, 6).map(([k, cnt]) => (
                                                                <Tag key={k}>{k}: {cnt}</Tag>
                                                            ))}
                                                            {Object.keys(cc || {}).length > 6 && <Tag>+{Object.keys(cc).length - 6}종</Tag>}
                                                        </div>
                                                    ),
                                                },
                                                {
                                                    title: '관리',
                                                    key: 'actions',
                                                    align: 'center' as const,
                                                    render: (_: any, rec: GtSetSummary) => (
                                                        <div style={{ display: 'flex', gap: 6, justifyContent: 'center' }}>
                                                            <Button size='small' onClick={() => onLoadGtSetDetail(rec.name)}>
                                                                이미지 보기
                                                            </Button>
                                                            <Button
                                                                size='small'
                                                                type='primary'
                                                                onClick={() => {
                                                                    setSelectedGtSet(rec.name);
                                                                    setEvalMode('gt_set');
                                                                    setEvalModalTab('run');
                                                                }}
                                                            >
                                                                이 셋으로 평가
                                                            </Button>
                                                            <Button size='small' danger icon={<DeleteOutlined />} onClick={() => onDeleteGtSet(rec.name)} />
                                                        </div>
                                                    ),
                                                },
                                            ]}
                                        />
                                    </Card>

                                    {/* 3) 선택한 정답지 셋 이미지 갤러리 */}
                                    {gtDetailSetName && (
                                        <Card size='small' title={`🖼️ '${gtDetailSetName}' 정답지 이미지 목록 (${gtDetailItems.length}장)`}>
                                            <Spin spinning={gtDetailLoading}>
                                                {gtDetailItems.length === 0 ? (
                                                    <Text type='secondary'>이미지가 없습니다.</Text>
                                                ) : (
                                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(190px, 1fr))', gap: 12, maxHeight: 340, overflowY: 'auto' }}>
                                                        {gtDetailItems.map((item) => (
                                                            <Card
                                                                key={item.id}
                                                                size='small'
                                                                cover={item.thumbnail ? (
                                                                    <img
                                                                        alt={item.id}
                                                                        src={item.thumbnail}
                                                                        style={{ height: 120, objectFit: 'cover' }}
                                                                    />
                                                                ) : undefined}
                                                                actions={[
                                                                    <Button
                                                                        key='del'
                                                                        type='text'
                                                                        danger
                                                                        size='small'
                                                                        icon={<DeleteOutlined />}
                                                                        onClick={() => onDeleteGtItem(gtDetailSetName, item.id)}
                                                                    >
                                                                        제외
                                                                    </Button>,
                                                                ]}
                                                            >
                                                                <div style={{ fontSize: 12 }}>
                                                                    <Text strong>Job #{item.source_job_id} · F{item.source_frame}</Text>
                                                                    <div>박스 {item.box_count}개</div>
                                                                    <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap', marginTop: 4 }}>
                                                                        {Object.entries(item.class_summary || {}).slice(0, 3).map(([cn, cnt]) => (
                                                                            <Tag key={cn} style={{ fontSize: 10, margin: 0 }}>{cn}:{cnt}</Tag>
                                                                        ))}
                                                                    </div>
                                                                </div>
                                                            </Card>
                                                        ))}
                                                    </div>
                                                )}
                                            </Spin>
                                        </Card>
                                    )}
                                </div>
                            ),
                        },
                    ]}
                />
            </Modal>
        </>
    );
}
